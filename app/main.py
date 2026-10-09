"""FastAPI application used identically by native hosts and CPU containers."""
from contextlib import asynccontextmanager
import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.catalog import load_catalog
from app.settings import Settings, ServiceError
from app.model_manager import ModelManager
from app.pubmed import PubMedClient
from app.schemas import ChatRequest, ChatResponse
from app.chat_service import ChatService


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
            {'id': spec.id, 'object': 'model', 'repo_id': spec.repo_id, 'revision': spec.revision}
            for spec in catalog.values()]}

    @application.post('/v1/chat/completions', response_model=ChatResponse)
    async def chat(request: ChatRequest):
        if request.model not in catalog:
            raise ServiceError('unknown_model', f'Unknown model: {request.model}', 422)
        return await service.complete(request)

    return application

app = create_app()
