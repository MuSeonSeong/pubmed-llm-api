"""Streamable HTTP MCP with a fixed read-only PubMed tool allowlist."""
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import logging
import re
import anyio
from jsonschema import validate, ValidationError
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from app.settings import ServiceError

ALLOWED = ('pubmed_search_articles', 'pubmed_fetch_articles')
TIMEOUT = 30
log = logging.getLogger('app.pubmed')
# A real calendar-shaped date: year, optional month 1-12 and day 1-31, not glued to more digits.
DATE = re.compile(r'(?<!\d)[12]\d{3}(?:[/.-](?:1[0-2]|0?[1-9])(?:[/.-](?:3[01]|[12]\d|0?[1-9]))?)?(?!\d)')
YEAR = re.compile(r'(?<!\d)[12]\d{3}(?!\d)')
# The search schema says an empty bound disables the whole range; these mean "no limit" on one side.
OPEN_DATE = {'minDate': '1000', 'maxDate': '3000'}
# Names small models use for the date bounds (compared after lower-casing and dropping punctuation).
DATE_ALIASES = {
    'mindate': 'minDate', 'startdate': 'minDate', 'fromdate': 'minDate', 'datefrom': 'minDate', 'from': 'minDate',
    'since': 'minDate', 'after': 'minDate', 'start': 'minDate', 'startyear': 'minDate', 'fromyear': 'minDate',
    'maxdate': 'maxDate', 'enddate': 'maxDate', 'todate': 'maxDate', 'dateto': 'maxDate', 'to': 'maxDate',
    'until': 'maxDate', 'before': 'maxDate', 'end': 'maxDate', 'endyear': 'maxDate', 'toyear': 'maxDate',
}
YEAR_ALIASES = {'year', 'pubyear', 'publicationyear', 'publishedyear'}
# A PMID written on its own or with a "PMID" label; never digits taken out of PMCIDs, DOIs or trial ids.
PMID_ITEM = re.compile(r'(?i)\s*(?:pmid\s*[:#]?\s*)?(\d{1,9})\s*')
PMID_IN_TEXT = re.compile(r'(?i)(?<![\w./-])(?:pmid\s*[:#]?\s*)?(\d{5,9})(?![\w./-])')
PMID_LIST_ITEM = re.compile(r'(?i)\s*(?:pmid\s*[:#]?\s*)?\d{1,9}(?:\s*[,;]\s*(?:pmid\s*[:#]?\s*)?\d{1,9})+\s*')
MAX_INT_DIGITS = 15


def _normalize(text: str) -> str:
    return re.sub(r'[^0-9a-z]', '', text.lower())


def _pmid(item):
    """A PMID string for one list entry, or the entry unchanged so validation reports it."""
    if isinstance(item, int) and not isinstance(item, bool) and item > 0:
        return str(item)
    if isinstance(item, str):
        match = PMID_ITEM.fullmatch(item)
        if match:
            return match.group(1)
    return item


def _coerce(value, schema: dict, notes: list[str], path: str):
    """Bring one value to the schema's type where the intent is unambiguous; leave the rest to validation.

    Notes record where and what kind of repair happened, never the values, which can quote the user.
    """
    kind = schema.get('type')
    if kind == 'object':
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return value
        return repair_arguments(value, schema, notes, path + '.') if isinstance(value, dict) else value
    if kind == 'array':
        pmids = schema.get('items', {}).get('pattern') == '^\\d+$'
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                # PMIDs may come as "24393785, 37185680"; any other text stays one entry ("Clinical Trial").
                parsed = PMID_IN_TEXT.findall(value) or [value] if pmids else [value]
            value = parsed if isinstance(parsed, list) else [parsed]
            notes.append(f'{path}: text became a list')
        elif not isinstance(value, list):
            value = [value]
            notes.append(f'{path}: single value became a list')
        if pmids:
            expanded = []
            for item in value:  # "24393785, 37185680" written as one entry
                if isinstance(item, str) and PMID_LIST_ITEM.fullmatch(item):
                    expanded.extend(re.findall(r'\d+', re.sub(r'(?i)pmid', '', item)))
                else:
                    expanded.append(item)
            fixed = [_pmid(item) for item in expanded]
            if fixed != value:
                notes.append(f'{path}: PMID labels removed')
            return fixed
        return [_coerce(item, schema.get('items', {}), notes, path + '[]') for item in value]
    if kind == 'integer' and isinstance(value, (str, float)) and not isinstance(value, bool):
        text = str(value).strip()
        if re.fullmatch(r'-?\d{1,%d}(\.0+)?' % MAX_INT_DIGITS, text):
            notes.append(f'{path}: text became an integer')
            return int(text.split('.')[0])
    if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        notes.append(f'{path}: text became a boolean')
        return value.strip().lower() == 'true'
    if kind == 'string':
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
            notes.append(f'{path}: number became text')
        if isinstance(value, str) and 'enum' in schema and value not in schema['enum']:
            matches = [option for option in schema['enum'] if _normalize(option) == _normalize(value)]
            if len(matches) == 1:
                notes.append(f'{path}: matched an allowed value')
                return matches[0]
    return value


def _repair_dates(args: dict, notes: list[str], path: str) -> None:
    """Normalise a {minDate, maxDate} object in place; afterwards it is complete, invalid or empty."""
    for key in OPEN_DATE:
        value = args.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = args[key] = str(int(value))
        if value is None or (isinstance(value, str) and not value.strip()):
            args.pop(key, None)
            continue
        if not isinstance(value, str):
            continue
        years = YEAR.findall(value)
        if len(years) >= 2 and not DATE.fullmatch(value.strip()):  # "2020-2022", "2020~2022", "2020년부터 2022년"
            low, high = sorted((years[0], years[-1]))
            other = 'maxDate' if key == 'minDate' else 'minDate'
            args[key] = low if key == 'minDate' else high
            if not (isinstance(args.get(other), str) and args[other].strip()) and not isinstance(args.get(other), (int, float)):
                args[other] = high if key == 'minDate' else low  # the other end of the written range
            notes.append(f'{path}{key}: a written range became the date bounds')
            continue
        match = DATE.search(value)
        if match and match.group() != value:
            args[key] = match.group()  # "2020년", "2020-01-01까지"
            notes.append(f'{path}{key}: reduced to a date')
    present = [key for key in OPEN_DATE if key in args]
    if len(present) == 1:  # one bound given means the other side is open
        missing = next(key for key in OPEN_DATE if key not in args)
        args[missing] = OPEN_DATE[missing]
        notes.append(f'{path}{missing}: filled as an open bound')


def repair_arguments(arguments: dict, schema: dict, notes: list[str] | None = None, path: str = '') -> dict:
    """Fix the argument mistakes small models make before strict validation runs.

    Unambiguous repairs only: misnamed or flattened fields go where they belong, fields the schema
    forbids are dropped, empty values are removed and scalar types are coerced. Anything else is left
    for validation so the model is asked to correct it rather than having its request silently changed.
    """
    notes = [] if notes is None else notes
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))
    args = deepcopy(arguments) if isinstance(arguments, dict) else {}
    for key, prop in properties.items():  # a nested object sent as JSON text, e.g. dateRange: "{...}"
        if prop.get('type') == 'object' and isinstance(args.get(key), str):
            try:
                parsed = json.loads(args[key])
            except ValueError:
                continue
            if isinstance(parsed, dict):
                args[key] = parsed
                notes.append(f'{path}{key}: JSON text became an object')
    by_name = {}
    for name in properties:  # publication_type and publicationTypes are the same field to a small model
        by_name.setdefault(_normalize(name), name)
        by_name.setdefault(_normalize(name).rstrip('s'), name)
    nested_owner = {}
    for name, prop in properties.items():
        for inner in prop.get('properties', {}) if prop.get('type') == 'object' else ():
            nested_owner.setdefault(_normalize(inner), (name, inner))
    date_owner = next((name for name, prop in properties.items()
                       if set(OPEN_DATE) <= set(prop.get('properties', {}))), None)
    for key in [key for key in args if key not in properties]:
        normal = _normalize(key)
        if normal in by_name or normal.rstrip('s') in by_name:  # max_results, MaxResults -> maxResults
            args.setdefault(by_name.get(normal) or by_name[normal.rstrip('s')], args.pop(key))
            notes.append(f'{path}{key}: renamed to a parameter')
            continue
        target = nested_owner.get(normal)
        if target is None and date_owner and normal in DATE_ALIASES:
            target = (date_owner, DATE_ALIASES[normal])
        if target is None and date_owner and normal in YEAR_ALIASES:
            nested = args.get(date_owner) if isinstance(args.get(date_owner), dict) else {}
            value = args.pop(key)
            nested.setdefault('minDate', value)
            nested.setdefault('maxDate', value)
            args[date_owner] = nested
            notes.append(f'{path}{key}: became a one-year date range')
            continue
        if target is not None:  # e.g. minDate given beside the query instead of inside dateRange
            owner, inner = target
            nested = args.get(owner) if isinstance(args.get(owner), dict) else {}
            nested.setdefault(inner, args.pop(key))
            args[owner] = nested
            notes.append(f'{path}{key}: moved into {owner}')
    if schema.get('additionalProperties') is False:
        for key in [key for key in args if key not in properties]:
            args.pop(key)
            notes.append(f'{path}{key}: dropped (not a parameter)')
    if set(OPEN_DATE) <= set(properties):
        _repair_dates(args, notes, path)
    for key in list(args):
        value = args[key]
        if value is None or (value in ('', [], {}) and key not in required):
            args.pop(key)
            notes.append(f'{path}{key}: dropped (empty)')
            continue
        args[key] = _coerce(value, properties.get(key, {}), notes, path + key)
        nested = properties.get(key, {})
        if nested.get('type') == 'object' and isinstance(args[key], dict) and not (
                set(nested.get('required', [])) & set(args[key])):
            args.pop(key)  # nothing usable left, e.g. a dateRange of nulls: no filter rather than an invalid one
            notes.append(f'{path}{key}: dropped (no values)')
    return args


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
        notes = []
        try:
            arguments = repair_arguments(arguments, self._tools[name].inputSchema, notes)
        except Exception:  # repair is best effort; validation still decides
            log.warning('argument repair failed for %s; validating the original arguments', name)
        if notes:
            log.info('repaired %s arguments: %s', name, '; '.join(notes))
        try:
            validate(arguments, self._tools[name].inputSchema)
        except ValidationError as exc:
            path = '.'.join(str(part) for part in exc.absolute_path) or '(arguments)'
            raise ServiceError('invalid_tool_arguments', exc.message, 502, {'path': path}) from exc
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
