"""FastAPI application used identically by native hosts and CPU containers."""
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import threading
import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from app.catalog import load_catalog
from app.settings import Settings, ServiceError
from app.model_manager import ModelManager
from app.pubmed import PubMedClient
from app.schemas import ChatRequest, ChatResponse
from app.chat_service import ChatService
from app.documents import MAX_PDF_BYTES, extract_pdf_text

MAX_BODY_BYTES = 64 * 1024 * 1024


class EventStreamResponse(Response):
    """Server-sent events whose producer is cancelled as soon as the client disconnects."""
    media_type = 'text/event-stream'

    def __init__(self, produce: Callable[[Callable[[dict], None], threading.Event], Awaitable[None]]):
        super().__init__(headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        # The body length is unknown; a declared Content-Length would truncate the stream.
        self.raw_headers = [(key, value) for key, value in self.raw_headers if key != b'content-length']
        self.produce = produce

    async def __call__(self, scope, receive, send):
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
                    async for event in events:
                        data = 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
                        await send({'type': 'http.response.body', 'body': data.encode(), 'more_body': True})
                await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            except OSError:
                pass  # the client vanished mid-write
            finally:
                cancel.set()
                group.cancel_scope.cancel()


def create_app(settings: Settings | None = None, manager: ModelManager | None = None,
               pubmed: PubMedClient | None = None) -> FastAPI:
    settings = settings or Settings()
    catalog = load_catalog()
    manager = manager if manager is not None else ModelManager(settings, catalog)
    pubmed = pubmed if pubmed is not None else PubMedClient(settings.pubmed_mcp_url)
    service = ChatService(manager, pubmed)

    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            await pubmed.close()
            await anyio.to_thread.run_sync(manager.close)

    application = FastAPI(title='PubMed Local LLM API', version='0.1.0', lifespan=lifespan)

    @application.middleware('http')
    async def limit_body(request: Request, call_next):
        # Four 8 MB images as base64 JSON stay well below this; anything larger is refused unread.
        if int(request.headers.get('content-length') or 0) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={'error': {'code': 'request_too_large',
                                'message': 'Request body is too large', 'details': {}}})
        return await call_next(request)
    web_dir = Path(__file__).resolve().parent / 'static'
    application.mount('/static', StaticFiles(directory=web_dir), name='static')

    @application.get('/', include_in_schema=False)
    async def chat_page():
        return FileResponse(web_dir / 'index.html', headers={'Cache-Control': 'no-cache'})

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={'error': {
            'code': exc.code, 'message': exc.message, 'details': exc.details}})

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [{'loc': list(item['loc']), 'msg': item['msg'], 'type': item['type']} for item in exc.errors()]
        return JSONResponse(status_code=422, content={'error': {'code': 'invalid_request',
                            'message': 'Invalid request', 'details': {'errors': errors}}})

    @application.get('/health')
    async def health():
        inference, dependency = manager.status(), pubmed.status()
        return {'status': 'ok', 'inference': inference, 'pubmed': dependency,
                'ready_for_chat': all(inference['model_files'].values()),
                'ready_for_pubmed': dependency['status'] == 'available'}

    @application.get('/v1/models')
    async def models():
        return {'object': 'list', 'default': 'qwen3.5-2b', 'data': [
            {'id': spec.id, 'object': 'model', 'repo_id': spec.repo_id, 'revision': spec.revision,
             'vision': spec.vision}
            for spec in catalog.values()]}

    @application.post('/v1/documents/extract')
    async def extract_document(request: Request):
        """Send a PDF as the raw request body (Content-Type: application/pdf) to get its text."""
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_PDF_BYTES:  # stop reading instead of buffering an unbounded upload
                raise ServiceError('document_too_large', 'PDF files must be 20 MB or smaller', 413)
        return await anyio.to_thread.run_sync(extract_pdf_text, bytes(body))

    @application.post('/v1/chat/completions', response_model=ChatResponse)
    async def chat(request: ChatRequest):
        if request.model not in catalog:
            raise ServiceError('unknown_model', f'Unknown model: {request.model}', 422)
        if request.has_images and not catalog[request.model].vision:
            raise ServiceError('vision_unsupported', f'{request.model} cannot read images; use qwen3.5-2b', 422)
        if request.stream:
            return EventStreamResponse(lambda emit, cancel: service.stream(request, emit, cancel))
        return await service.complete(request)

    return application

app = create_app()
