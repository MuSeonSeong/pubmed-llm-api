"""FastAPI application used identically by native hosts and CPU containers."""
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import base64
import json
import logging
import math
from pathlib import Path
import threading
from typing import Annotated
import anyio
import fastapi
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from app.auth import PUBLIC_API, Auth, User, unauthorized
from app.catalog import ModelSpec, load_catalog
from app.settings import Settings, ServiceError, enabled_models
from app.model_manager import THINKING_BUDGETS, ModelManager
from app.pubmed import PubMedClient
from app.schemas import CHAT_ID, ChatRequest, ChatResponse, LoginRequest
from app.chat_service import GUIDANCE, ChatService, error_payload
from app.documents import MAX_PDF_BYTES, extract_pdf_text
from app.images import MAX_BYTES as MAX_IMAGE_BYTES, invalid, validate_image_bytes
from app.storage import LOCAL_USER, StaleChat, Storage

MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_CHAT_BYTES = 4 * 1024 * 1024
MAX_SETTINGS_BYTES = 32 * 1024
MAX_REV = 2 ** 53 - 1  # the largest integer a browser keeps exact
MAX_SEQ = 2 ** 63 - 1
COMPLETIONS_PER_USER = 2
HEARTBEAT_SECONDS = 15  # proxies such as Cloudflare close a response that stays silent for 100 s
FILE_URL = '/api/files/'
UPLOAD_TYPES = frozenset({'image/png', 'image/jpeg', 'image/webp', 'image/gif'})
WEB_DIR = Path(__file__).resolve().parent / 'static'
SECURITY_HEADERS = {'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'same-origin'}
# The pages run only their own scripts and styles, so markup that slips into a chat cannot execute.
PAGE_HEADERS = {
    'Cache-Control': 'no-cache', 'X-Frame-Options': 'DENY',
    'Content-Security-Policy': ("default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
                                "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                                "frame-ancestors 'none'; form-action 'self'")}
ChatId = Annotated[str, fastapi.Path(pattern=CHAT_ID)]


class EventStreamResponse(Response):
    """Server-sent events whose producer is cancelled as soon as the client disconnects.

    on_close runs once the stream is over, whether it finished or the client left.
    """
    media_type = 'text/event-stream'

    def __init__(self, produce: Callable[[Callable[[dict], None], threading.Event], Awaitable[None]],
                 on_close: Callable[[], None] | None = None):
        super().__init__(headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        # The body length is unknown; a declared Content-Length would truncate the stream.
        self.raw_headers = [(key, value) for key, value in self.raw_headers if key != b'content-length']
        self.produce, self.on_close = produce, on_close

    async def __call__(self, scope, receive, send):
        try:
            await self._stream(receive, send)
        finally:
            if self.on_close is not None:
                self.on_close()

    async def _stream(self, receive, send):
        cancel = threading.Event()
        sender, events = anyio.create_memory_object_stream(math.inf)

        async def run():
            with sender:
                await self.produce(sender.send_nowait, cancel)

        async def watch_disconnect():
            while (await receive())['type'] != 'http.disconnect':
                pass
            cancel.set()
            group.cancel_scope.cancel()

        await send({'type': 'http.response.start', 'status': 200, 'headers': self.raw_headers})
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            group.start_soon(watch_disconnect)
            try:
                async with events:
                    while True:
                        event = None
                        with anyio.move_on_after(HEARTBEAT_SECONDS):
                            try:
                                event = await events.receive()
                            except anyio.EndOfStream:
                                break
                        # A comment line keeps the connection open while a model loads or a tool runs.
                        data = ': ping' if event is None else 'data: ' + json.dumps(event, ensure_ascii=False)
                        await send({'type': 'http.response.body', 'body': (data + '\n\n').encode(), 'more_body': True})
                await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            except OSError:
                pass  # the client vanished mid-write
            finally:
                cancel.set()
                group.cancel_scope.cancel()


class Activity:
    """Completions in progress per user and the chats they answer; used only on the event loop."""

    def __init__(self, limit: int):
        self.limit = limit
        self.running: Counter[int] = Counter()
        self.chats: defaultdict[int, Counter[str]] = defaultdict(Counter)

    def start(self, user_id: int, chat_id: str | None = None) -> Callable[[], None]:
        """Take one of the user's slots and return the call that gives it back."""
        if self.running[user_id] >= self.limit:
            raise ServiceError('busy', f'At most {self.limit} answers per user can be generated at once', 429)
        self.running[user_id] += 1
        if chat_id:
            self.chats[user_id][chat_id] += 1
        finished = False

        def finish() -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            self.running[user_id] -= 1
            if self.running[user_id] <= 0:
                del self.running[user_id]
            if chat_id:
                live = self.chats[user_id]
                live[chat_id] -= 1
                if live[chat_id] <= 0:
                    del live[chat_id]
                if not live:
                    del self.chats[user_id]
        return finish

    def live(self, user_id: int) -> list[str]:
        """Chats being answered right now, so other devices can show them as in progress."""
        return sorted(self.chats.get(user_id, ()))


def model_state(status: dict, model_id: str) -> str:
    """missing, stopped, starting, ready or failed; a manager without per-model states reports files only."""
    info = (status.get('models') or {}).get(model_id)
    if info:
        return info['state']
    return 'ready' if status.get('model_files', {}).get(model_id) else 'missing'


def model_card(spec: ModelSpec, state: str, settings: Settings) -> dict:
    return {'id': spec.id, 'object': 'model', 'repo_id': spec.repo_id, 'revision': spec.revision,
            'label': spec.label, 'summary': spec.summary, 'vision': spec.vision, 'reasoning': spec.reasoning,
            'efforts': list(THINKING_BUDGETS) if spec.reasoning else [],
            'context_tokens': spec.context_tokens or settings.max_context_tokens, 'state': state}


def error_response(exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, headers=exc.headers, content={'error': error_payload(exc)})


def raw_json(text: str, status_code: int = 200) -> Response:
    return Response(content=text, status_code=status_code, media_type='application/json')


async def read_body(request: Request, limit: int, too_large: ServiceError) -> bytes:
    """The raw body, refused as soon as it grows past `limit` instead of buffering all of it."""
    if int(request.headers.get('content-length') or 0) > limit:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_large
    return bytes(body)


def _reject_constant(name: str):
    raise ValueError(f'{name} is not valid JSON')


def parse_object(body: bytes, code: str) -> tuple[str, dict]:
    """The body as text and as a parsed object.

    The text is stored as sent and later spliced into responses unparsed, so it must be strict JSON:
    Python alone would accept NaN, which the browser's JSON.parse rejects.
    """
    try:
        text = body.decode('utf-8')
        value = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ServiceError(code, 'The body must be a JSON object', 422) from exc
    if not isinstance(value, dict):
        raise ServiceError(code, 'The body must be a JSON object', 422)
    return text, value


def create_app(settings: Settings | None = None, manager: ModelManager | None = None,
               pubmed: PubMedClient | None = None) -> FastAPI:
    settings = settings or Settings()
    catalog = load_catalog()
    enabled = enabled_models(settings, catalog)  # a wrong MODELS stops the server here
    manager = manager if manager is not None else ModelManager(settings, catalog)
    pubmed = pubmed if pubmed is not None else PubMedClient(settings.pubmed_mcp_url)
    service = ChatService(manager, pubmed)
    storage = Storage(settings.data_dir, settings.user_storage_mb * 1024 * 1024)
    auth = Auth(settings, storage)
    activity = Activity(COMPLETIONS_PER_USER)
    web_dir = WEB_DIR

    async def pubmed_prefix() -> tuple[str, list[dict]]:
        return GUIDANCE, await pubmed.tool_schemas()

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with anyio.create_task_group() as group:
                # MLX models load in the background, so pages and /health answer meanwhile, and the manager owns
                # every model start until shutdown cancels it. With accounts every user's prompts are cached
                # apart, so only local mode gains from caching the PubMed prompt after a start.
                group.start_soon(manager.start, None if settings.auth_enabled else pubmed_prefix)
                yield
                group.cancel_scope.cancel()
        finally:
            await pubmed.close()
            await anyio.to_thread.run_sync(manager.close)
            await anyio.to_thread.run_sync(storage.close)

    # With accounts, the API description is not published to whoever finds the address.
    docs = {'openapi_url': None} if settings.auth_enabled else {}
    application = FastAPI(title='PubMed Local LLM API', version='0.1.0', lifespan=lifespan, **docs)
    application.state.storage = storage
    signed_in = Depends(auth.current_user)

    async def screen(request: Request) -> Response | None:
        """Refuse a request before any route reads its body."""
        # Four 8 MB images as base64 JSON stay well below this; anything larger is refused unread.
        if int(request.headers.get('content-length') or 0) > MAX_BODY_BYTES:
            return error_response(ServiceError('request_too_large', 'Request body is too large', 413))
        # Browsers label requests that other sites make; none of those may change anything here.
        if (request.method not in ('GET', 'HEAD', 'OPTIONS')
                and request.headers.get('sec-fetch-site') in ('cross-site', 'same-site')):
            return error_response(ServiceError('cross_site_request', 'Requests from other sites are refused', 403))
        path = request.url.path
        if settings.auth_enabled and path.startswith(('/api/', '/v1/')) and path not in PUBLIC_API:
            user = await auth.session_user(request)
            if user is None:
                return error_response(unauthorized())
            request.state.user = user  # read by auth.current_user without a second lookup
        return None

    @application.middleware('http')
    async def guard(request: Request, call_next):
        response = await screen(request)
        if response is None:
            try:
                response = await call_next(request)
            except Exception:  # still the JSON error shape and the headers below, never a bare 500 page
                logging.getLogger('uvicorn.error').exception('Unhandled error in %s %s', request.method,
                                                             request.url.path)
                response = error_response(ServiceError('internal_error', 'Internal error', 500))
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        path = request.url.path
        if path.startswith('/api/'):
            response.headers.setdefault('Cache-Control', 'no-store')  # private data, never kept by proxies
        elif path.startswith('/static/'):
            # Revalidate on every load (a cheap 304) so an update reaches browsers behind a proxy at once.
            response.headers.setdefault('Cache-Control', 'no-cache')
        return response
    application.mount('/static', StaticFiles(directory=web_dir), name='static')

    @application.get('/', include_in_schema=False)
    async def chat_page(request: Request):
        if settings.auth_enabled and await auth.session_user(request) is None:
            return RedirectResponse('/login', status_code=302, headers={'Cache-Control': 'no-cache'})
        return FileResponse(web_dir / 'index.html', headers=PAGE_HEADERS)

    @application.get('/login', include_in_schema=False)
    async def login_page(request: Request):
        if not settings.auth_enabled or await auth.session_user(request) is not None:
            return RedirectResponse('/', status_code=302, headers={'Cache-Control': 'no-cache'})
        return FileResponse(web_dir / 'login.html', headers=PAGE_HEADERS)

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(exc)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [{'loc': list(item['loc']), 'msg': item['msg'], 'type': item['type']} for item in exc.errors()]
        return JSONResponse(status_code=422, content={'error': {'code': 'invalid_request',
                            'message': 'Invalid request', 'details': {'errors': errors}}})

    @application.get('/health')
    async def health(request: Request):
        # Anyone may learn that the server is up; what it runs is for signed-in users.
        if settings.auth_enabled and await auth.session_user(request) is None:
            return {'status': 'ok'}
        inference, dependency = manager.status(), pubmed.status()
        return {'status': 'ok', 'inference': inference, 'pubmed': dependency,
                'ready_for_chat': model_state(inference, enabled[0]) == 'ready',
                'ready_for_pubmed': dependency['status'] == 'available'}

    @application.get('/api/session')
    async def session(request: Request):
        if not settings.auth_enabled:
            return {'auth_enabled': False, 'user': {'name': LOCAL_USER}}
        user = await auth.session_user(request)
        return {'auth_enabled': True, 'user': {'name': user.name} if user else None}

    @application.post('/api/login')
    async def login(body: LoginRequest, request: Request):
        if not settings.auth_enabled:
            raise ServiceError('auth_disabled', 'Sign-in is off; the server runs in local mode', 400)
        user, token = await auth.login(body.name, body.password, request.client.host if request.client else '')
        response = JSONResponse({'user': {'name': user.name}})
        auth.set_cookie(response, request, token)
        return response

    @application.post('/api/logout', status_code=204)
    async def logout(request: Request):
        await auth.logout(request)
        response = Response(status_code=204)
        auth.clear_cookie(response, request)
        return response

    @application.get('/api/chats')
    async def chat_changes(since: Annotated[int, Query(ge=0, le=MAX_SEQ)] = 0, user: User = signed_in):
        """Chats changed and deleted after `since`; pass the returned seq back as `since` next time."""
        seq, chats, deleted = await anyio.to_thread.run_sync(storage.changes, user.id, since)
        # Stored chats are spliced in as the JSON text the client sent instead of being parsed again.
        return raw_json('{"seq":%d,"chats":[%s],"deleted":%s,"live":%s}' % (
            seq, ','.join(chats), json.dumps(deleted), json.dumps(activity.live(user.id))))

    @application.put('/api/chats/{chat_id}')
    async def save_chat(chat_id: ChatId, request: Request, user: User = signed_in):
        """The body is the chat object; it is kept unless the server already has a newer revision."""
        body = await read_body(request, MAX_CHAT_BYTES,
                               ServiceError('chat_too_large', 'Chats must be 4 MB or smaller', 413))

        def save():
            text, chat = parse_object(body, 'invalid_chat')
            rev = chat.get('rev')
            if chat.get('id') != chat_id or type(rev) is not int or not 0 <= rev <= MAX_REV:
                raise ServiceError('invalid_chat', 'A chat needs the id from its URL and an integer rev of 0 or more',
                                   422)
            return storage.put_chat(user.id, chat_id, rev, text)
        try:
            rev, seq = await anyio.to_thread.run_sync(save)
        except StaleChat as stale:
            error = error_payload(ServiceError('stale_chat', 'The server has a newer revision of this chat', 409))
            return raw_json('{"error":%s,"chat":%s}' % (json.dumps(error), stale.data), 409)
        return {'rev': rev, 'seq': seq}

    @application.delete('/api/chats/{chat_id}', status_code=204)
    async def delete_chat(chat_id: ChatId, user: User = signed_in):
        await anyio.to_thread.run_sync(storage.delete_chat, user.id, chat_id)
        return Response(status_code=204)

    @application.get('/api/settings')
    async def get_settings(user: User = signed_in):
        data = await anyio.to_thread.run_sync(storage.settings, user.id)
        return raw_json('{"settings":%s}' % (data or 'null'))

    @application.put('/api/settings', status_code=204)
    async def put_settings(request: Request, user: User = signed_in):
        body = await read_body(request, MAX_SETTINGS_BYTES,
                               ServiceError('settings_too_large', 'Settings must be 32 KB or smaller', 413))
        text, _ = parse_object(body, 'invalid_settings')
        await anyio.to_thread.run_sync(storage.put_settings, user.id, text)
        return Response(status_code=204)

    @application.post('/api/files', status_code=201)
    async def upload_file(request: Request, user: User = signed_in):
        """Send a PNG, JPEG, WebP or GIF image as the raw body; its type is read from the bytes."""
        declared = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
        if declared not in UPLOAD_TYPES:
            raise ServiceError('unsupported_media_type', 'Upload a PNG, JPEG, WebP or GIF image', 415)
        body = await read_body(request, MAX_IMAGE_BYTES,
                               ServiceError('file_too_large', 'Images must be 8 MB or smaller', 413))

        def store():
            mime = validate_image_bytes(body)
            return mime, storage.add_file(user.id, mime, body)
        mime, file_id = await anyio.to_thread.run_sync(store)
        return JSONResponse(status_code=201, content={'id': file_id, 'url': FILE_URL + file_id, 'mime': mime,
                                                      'size': len(body)})

    @application.get('/api/files/{file_id}')
    async def download_file(file_id: str, user: User = signed_in):
        found = await anyio.to_thread.run_sync(storage.find_file, user.id, file_id)
        if found is None:  # someone else's file looks exactly like a missing one
            raise ServiceError('not_found', 'File not found', 404)
        mime, path = found
        return FileResponse(path, media_type=mime, headers={
            'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'private, max-age=31536000, immutable'})

    def file_data_url(user_id: int, file_id: str) -> str:
        found = storage.read_file(user_id, file_id)
        if found is None:
            raise invalid('Image not found')
        mime, data = found
        return f'data:{mime};base64,' + base64.b64encode(data).decode()

    async def cache_tenant(user: User) -> str | None:
        """With accounts, the MLX prefix cache is salted per account, so nobody can time another's prompts.

        An id alone is not enough: SQLite gives a removed newest account's id to the next account created.
        """
        if not settings.auth_enabled:
            return None
        created = await anyio.to_thread.run_sync(storage.user_created, user.id)
        if created is None:  # removed since this request signed in
            raise unauthorized()
        return f'user-{user.id}-{created}'

    async def resolve_files(user: User, request: ChatRequest) -> None:
        """Swap the caller's own uploads (/api/files/<id>) for data URLs the model manager decodes."""
        for message in request.messages:
            for part in message.content if isinstance(message.content, list) else []:
                url = part.image_url.url.strip() if part.type == 'image_url' else ''
                if url.startswith(FILE_URL):
                    part.image_url.url = await anyio.to_thread.run_sync(file_data_url, user.id, url[len(FILE_URL):])

    @application.get('/v1/models', dependencies=[signed_in])
    async def models():
        """The enabled models, default first, with what each can do and whether it is ready."""
        status = manager.status()
        return {'object': 'list', 'default': enabled[0], 'data': [
            model_card(catalog[model_id], model_state(status, model_id), settings) for model_id in enabled]}

    @application.post('/v1/documents/extract', dependencies=[signed_in])
    async def extract_document(request: Request):
        """Send a PDF as the raw request body (Content-Type: application/pdf) to get its text."""
        body = await read_body(request, MAX_PDF_BYTES,
                               ServiceError('document_too_large', 'PDF files must be 20 MB or smaller', 413))
        return await anyio.to_thread.run_sync(extract_pdf_text, body)

    @application.post('/v1/chat/completions', response_model=ChatResponse)
    async def chat(request: ChatRequest, user: User = signed_in):
        request.model = request.model or enabled[0]
        if request.model not in enabled:
            raise ServiceError('unknown_model', f'Unknown model: {request.model}', 422)
        spec = catalog[request.model]
        if request.has_images and not spec.vision:
            raise ServiceError('vision_unsupported', f'{request.model} cannot read images; choose a model with '
                               'vision in /v1/models', 422)
        if not spec.reasoning:
            request.reasoning_effort = 'none'
        tenant = await cache_tenant(user)
        finish = activity.start(user.id, request.chat_id if request.stream else None)
        try:
            await resolve_files(user, request)
            if request.stream:
                response = EventStreamResponse(lambda emit, cancel: service.stream(request, emit, cancel, tenant),
                                               on_close=finish)
                finish = None  # released by the response once the stream is over
                return response
            return await service.complete(request, tenant)
        finally:
            if finish is not None:
                finish()

    return application

app = create_app()
