"""Streamable HTTP MCP with a fixed read-only PubMed tool allowlist."""
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import anyio
from jsonschema import validate, ValidationError
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from app.settings import ServiceError

ALLOWED = ('pubmed_search_articles', 'pubmed_fetch_articles')
TIMEOUT = 30


def compact_schema(value):
    if isinstance(value, dict):
        return {key: (item[:120] if key == 'description' and isinstance(item, str) else compact_schema(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    return value


def format_tool_result(result: dict) -> str:
    text = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    if len(text) <= 12000:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = json.dumps({'truncated': True, 'preview': text[:mid]}, ensure_ascii=False)
        if len(candidate) <= 12000:
            low = mid
        else:
            high = mid - 1
    return json.dumps({'truncated': True, 'preview': text[:low]}, ensure_ascii=False)


def has_timeout(exc) -> bool:
    return isinstance(exc, TimeoutError) or any(has_timeout(item) for item in getattr(exc, 'exceptions', []))

class PubMedClient:
    def __init__(self, url: str):
        self.url = url
        self._tools = {}
        self._state = {'status': 'unknown', 'error': None}

    def status(self) -> dict:
        return dict(self._state)

    async def close(self) -> None:
        # Sessions are owned and closed by the operation that opens them.
        pass

    @asynccontextmanager
    async def _session(self):
        async with streamablehttp_client(self.url, timeout=TIMEOUT, sse_read_timeout=TIMEOUT) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def _error(self, exc: Exception):
        self._state = {'status': 'unavailable', 'error': str(exc)}
        if isinstance(exc, ServiceError):
            raise exc
        timeout = has_timeout(exc)
        raise ServiceError('pubmed_timeout' if timeout else 'pubmed_unavailable',
                           str(exc) or 'PubMed MCP request failed', 504 if timeout else 502) from exc

    async def tool_schemas(self) -> list[dict]:
        if not self._tools:
            try:
                with anyio.fail_after(TIMEOUT):
                    async with self._session() as session:
                        response = await session.list_tools()
                        self._tools = {tool.name: tool for tool in response.tools if tool.name in ALLOWED}
                        if set(self._tools) != set(ALLOWED):
                            raise ServiceError('pubmed_schema_missing', 'Required PubMed tools are unavailable', 502)
                        self._state = {'status': 'available', 'error': None}
            except Exception as exc:
                self._tools = {}
                self._error(exc)
        schemas = []
        for name in ALLOWED:
            tool = self._tools[name]
            params = compact_schema(deepcopy(tool.inputSchema))
            props = params['properties']
            if name == ALLOWED[0]:
                props['maxResults'].update(maximum=5, default=5)
                props['summaryCount'].update(maximum=5, default=3)
            else:
                props['pmids']['maxItems'] = 3
                props['includeMesh']['default'] = False
                props['maxResponseCharacters']['maximum'] = 10000
            schemas.append({'type': 'function', 'function': {'name': name,
                'description': (tool.description or '')[:240], 'parameters': params}})
        return schemas

    async def validate_call(self, name: str, arguments: dict) -> dict:
        if name not in ALLOWED:
            raise ServiceError('unknown_tool', f'Tool is not allowed: {name}', 502)
        await self.tool_schemas()
        try:
            validate(arguments, self._tools[name].inputSchema)
        except ValidationError as exc:
            raise ServiceError('invalid_tool_arguments', exc.message, 502) from exc
        args = deepcopy(arguments)
        if name == ALLOWED[0]:
            args['maxResults'] = min(args.get('maxResults', 5), 5)
            args['summaryCount'] = min(args.get('summaryCount', 3), args['maxResults'], 5)
        else:
            args['pmids'] = args['pmids'][:3]
            args.setdefault('includeMesh', False)
            args.setdefault('includeGrants', False)
            args['maxResponseCharacters'] = min(args.get('maxResponseCharacters', 10000), 10000)
        return args

    async def call(self, name: str, arguments: dict) -> dict:
        args = await self.validate_call(name, arguments)
        try:
            with anyio.fail_after(TIMEOUT):
                async with self._session() as session:
                    response = await session.call_tool(name, args)
            result = response.structuredContent
            if result is None:
                text = '\n'.join(part.text for part in response.content if part.type == 'text')
                try:
                    result = json.loads(text)
                except ValueError:
                    result = {'text': text}
            if not isinstance(result, dict):
                result = {'data': result}
            if response.isError or result.get('error'):
                raise ServiceError('pubmed_tool_failed', 'PubMed returned a tool error', 502, {'result': result})
            self._state = {'status': 'available', 'error': None}
            return result
        except Exception as exc:
            self._error(exc)
