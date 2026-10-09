# PubMed Transformers API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 다른 PC에서도 사용하는 Transformers 기반 채팅·PubMed API와 모델 2개, 로컬 `.venv`, Docker 실행 환경을 완성한다.

**Architecture:** FastAPI 단일 프로세스가 Transformers 생성과 PubMed MCP를 조합한다. 모델은 필요할 때 하나만 로드하고 생성 작업은 직렬화한다. 동일한 Python 소스로 CUDA/MPS/CPU를 선택하며 Docker 기본 구성은 CPU다.

**Tech Stack:** Python 3.12, uv, PyTorch, Transformers, Accelerate, Safetensors, FastAPI, Uvicorn, httpx, MCP Python SDK, jsonschema, pytest, AnyIO, Docker Compose.

**Spec:** [승인된 설계](../specs/2026-10-09-pubmed-llm-api-design.md)

## Global Constraints

- Python 3.12, 프로젝트 전용 `.venv`, `pyproject.toml`, `uv.lock`.
- `qwen3-0.6b`: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- 기본 모델 `qwen3.5-2b`: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- 저장 위치 `models/qwen3-0.6b/`, `models/qwen3.5-2b/`. 원본 캐시를 수정하지 않고 외부 심볼릭 링크를 남기지 않는다.
- `DEVICE=auto|cuda|mps|cpu`; auto 순서 CUDA → MPS → CPU. CUDA BF16 지원 시 BF16/그 외 FP16, MPS FP16, CPU FP32.
- FastAPI/Uvicorn 포트 8000, worker 1개, 모델 1개, 생성 직렬화, 요청별 독립 생성 캐시.
- `max_tokens` 기본 512·최대 2048, `temperature` 기본 0.2·범위 0~2, `use_pubmed` 기본 true.
- `MAX_CONTEXT_TOKENS` 기본 4096·상한 8192. 실제 템플릿 입력 + 생성 예약 토큰을 매번 검사한다.
- `PUBMED_MCP_URL` 기본 `https://pubmed.caseyjhand.com/mcp`; Streamable HTTP; 연결·개별 호출 제한 30초.
- 도구는 `pubmed_search_articles`, `pubmed_fetch_articles`만. 검색 최대 5건, 상세 최대 3건, 호출별 결과 최대 12,000자, 요청당 도구 실행 최대 4회.
- `.venv`, 모델, 캐시, 로컬 환경 파일을 Git/Docker 이미지에서 제외한다. MLX·FlashAttention·bitsandbytes는 필수 의존성에 넣지 않는다.
- macOS·Windows·Linux는 같은 소스를 사용한다. 실제 실행하지 않은 플랫폼은 검증 완료로 표시하지 않는다.
- 원격 Git 저장소, 웹 UI, 계정, DB, 원격 배포, NVIDIA용 Docker 변형은 이번 범위에 추가하지 않는다.

## Review Focus

1. 프로젝트 경로에 공백이 있거나 캐시가 심볼릭 링크일 때, 모델을 독립적인 파일로 준비하고 불완전한 다운로드를 완료로 취급하지 않아야 한다. Task 1의 파일 준비 테스트.
2. 요청 취소 중 생성이 계속될 때, 다음 요청이 동시에 생성하거나 모델을 해제하면 안 된다. Task 2의 취소·전환 테스트.
3. PubMed 응답이 지나치게 크거나 도구 결과 추가 후 문맥 한도를 넘을 때, 잘림을 표시하거나 명확히 거절해야 한다. Tasks 3/4의 결과 크기·문맥 테스트.
4. 한 응답에 도구 호출이 여러 개 있거나 잘못된 호출이 섞였을 때, 총 4회 제한을 넘기거나 검증 전 외부 도구를 호출하면 안 된다. Task 4의 전체 사전 검증 테스트.
5. PubMed 서버가 꺼져 있을 때, 도구 없는 채팅과 모델 목록은 동작하고 도구 채팅은 성공으로 위장하지 않아야 한다. Tasks 3/4의 장애·API 테스트.

---

## 파일 책임

- `app/catalog.py`, `models.json`: 모델 ID, 원본·리비전·상대 디렉터리·로더 종류.
- `app/settings.py`, `app/schemas.py`: 환경 설정, 요청/응답 계약 및 공통 오류.
- `scripts/prepare_models.py`: 캐시 활용·다운로드·완료 확인.
- `app/model_manager.py`: 장치 선택, 하나의 모델 로드, 토큰 예산 검사, 직렬 생성.
- `app/pubmed.py`: MCP 세션과 도구 스키마·호출 제한·결과 정리.
- `app/chat_service.py`: Qwen 도구 호출 파싱과 도구 실행 반복.
- `app/main.py`: 라우팅, 의존성 주입, lifespan, 오류를 HTTP 응답으로 변환.
- `tests/test_catalog.py`, `test_prepare_models.py`, `test_model_manager.py`, `test_pubmed.py`, `test_chat_service.py`, `test_api.py`: 아래의 동작별 테스트.
- `pyproject.toml`, `uv.lock`, `.python-version`, `.env.example`, `.gitignore`, `.dockerignore`: 재현 가능한 환경.
- `Dockerfile`, `compose.yaml`, `README.md`: CPU 컨테이너, 모델 볼륨, 실행 및 검증 기록.

## Task 1: 로컬 환경과 재현 가능한 모델 준비

**Files:** Create `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `.dockerignore`, `.env.example`, `models.json`, `app/__init__.py`, `app/catalog.py`, `app/settings.py`, `scripts/prepare_models.py`, `tests/test_catalog.py`, `tests/test_prepare_models.py`.

**Interfaces:**
- `ModelSpec(id: str, repo_id: str, revision: str, directory: str, loader: str)` immutable dataclass. loader는 `causal_lm` 또는 `qwen3_5`.
- `load_catalog() -> dict[str, ModelSpec]`: 패키지 기준 프로젝트 매니페스트를 읽고 정확히 2개를 반환한다.
- `Settings`: `device`, `models_dir: Path`, `max_context_tokens: int`, `pubmed_mcp_url: str`; 기본값은 Global Constraints. 상대 경로는 프로젝트 루트 기준으로 해석한다.
- `prepare_model(spec: ModelSpec, destination: Path) -> Path`: 기존 HF 캐시 활용 후 고정 리비전을 준비하고 필수 파일을 검증한다. CLI는 `uv run python scripts/prepare_models.py`.
- 공통 `ServiceError(code: str, message: str, status_code: int, details: dict | None = None)`는 `app/settings.py`에 정의하며 이후 작업에서 재사용한다.

- [x] **Step 1:** `.python-version`, 기본 패키지/pytest 설정과 제외 규칙을 작성한다. Torch CPU 배포 인덱스는 Windows/Linux에만 적용하고 macOS는 PyPI 배포를 쓴다. Transformers는 Qwen3.5를 지원하는 설치된 안정 버전 이상으로 고정한다. `uv lock`, `uv sync --dev`로 새 `.venv`를 만든다. 구성 파일에 의미 없는 TDD 테스트를 붙이지 않는다.
- [x] **Step 2:** 실패할 테스트를 작성한다. 아래 assertion과 파일 준비 조건을 포함한다.

```python
assert set(load_catalog()) == {"qwen3-0.6b", "qwen3.5-2b"}
assert load_catalog()["qwen3.5-2b"].revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
assert Settings().max_context_tokens == 4096
```

`test_cache_symlinks_become_independent_files`는 공백 있는 임시 경로와 가짜 캐시 심볼릭 링크를 만들고, 준비 후 원본을 제거해도 목적지 파일 내용을 읽을 수 있고 링크가 없음을 확인한다. `test_partial_download_is_not_complete`는 가중치 누락·중단 시 완료 표시가 생기지 않음을 확인한다. `test_existing_complete_model_skips_download`는 완료된 디렉터리에서 네트워크 다운로드를 호출하지 않음을 확인한다.

- [x] **Step 3:** `uv run pytest tests/test_catalog.py tests/test_prepare_models.py -q` 실행. 미구현 인터페이스로 실패하는 것을 확인한다.
- [x] **Step 4:** 위 인터페이스를 구현한다. HF 캐시 탐색에 사용자 홈과 표준 설정을 사용한다. 다운로드는 `snapshot_download`의 고정 리비전과 필요한 모델/토크나이저 파일만 사용한다. 복사·완료 검증 후에만 완료 표시를 쓴다. 모델 경로 누락은 실행 가능한 준비 명령을 포함한 오류로 표현한다.
- [x] **Step 5:** 같은 테스트를 실행해 통과를 확인하고 `uv run python scripts/prepare_models.py`로 실제 두 모델을 준비한다. 실제 파일, 리비전, 외부 링크가 없는 것을 확인한다.
- [x] **Step 6:** 이 Task의 소스·환경 파일·테스트만 Git에 추가하고 `git commit -m "feat: prepare portable environment and model files"`. `.venv`와 가중치가 추적되지 않는지 확인한다.

## Task 2: Transformers 모델 관리와 직렬 추론

**Files:** Create `app/model_manager.py`, `tests/test_model_manager.py`. Consumes Task 1.

**Interfaces:**
- `select_device(requested: str, *, cuda_available: bool, mps_available: bool) -> str`: 장치 선택, 없는 명시 장치는 `ServiceError`.
- `GenerationResult(raw_text: str, finish_reason: str, prompt_tokens: int, completion_tokens: int)` dataclass.
- `ModelManager(settings: Settings, catalog: dict[str, ModelSpec])`.
- `await manager.generate(model_id: str, messages: list[dict], tools: list[dict], max_tokens: int, temperature: float) -> GenerationResult`.
- `manager.status() -> dict`: 선택 장치, 현재 모델 ID, 파일 존재 상태. `manager.close() -> None`: 모델 해제.

- [x] **Step 1:** 실패할 테스트를 작성한다.

```python
assert select_device("auto", cuda_available=True, mps_available=True) == "cuda"
assert select_device("auto", cuda_available=False, mps_available=True) == "mps"
assert select_device("auto", cuda_available=False, mps_available=False) == "cpu"
```

`test_explicit_unavailable_device_fails`는 가용하지 않은 CUDA/MPS 선택에 오류를 확인한다. `test_context_counts_template_and_tools`는 입력+생성이 4096을 넘으면 생성 호출 전에 거절하고 경계값은 허용함을 확인한다. `test_model_switch_releases_previous_model`은 모델 교체 순서를 확인한다. `test_cancelled_request_keeps_generation_serialized`는 첫 생성이 대기 중 취소되어도 두 번째 생성·모델 교체가 첫 생성 종료 전 시작되지 않음을 이벤트로 검사한다. `test_raw_generation_preserves_tool_markers`는 도구 마커 유지와 프롬프트 제외를 검사한다. 장치/동시성 테스트는 가짜 작은 백엔드를 쓰며 실제 모델 검증을 대신하지 않는다.

- [x] **Step 2:** `uv run pytest tests/test_model_manager.py -q` 실행, 미구현으로 실패 확인.
- [x] **Step 3:** 모델 관리 인터페이스를 구현한다. CPU/GPU dtype은 Global Constraints를 적용한다. Qwen3는 AutoModelForCausalLM, Qwen3.5는 공식 Qwen3.5 로더를 사용한다. 플랫폼 전용 커널 없이 기본 PyTorch 경로를 사용한다. actual chat template에 `enable_thinking=False`, 도구 스키마, `return_dict=True`를 적용한다. 생성 부분만 `skip_special_tokens=False`로 디코드해 반환한다.
- [x] **Step 4:** 모델 로드·해제·생성 전체를 같은 스레드 잠금으로 보호한다. 블로킹 생성은 AnyIO worker thread에서 실행해 이벤트 루프를 막지 않는다. HTTP 코루틴 취소와 무관하게 실제 worker의 생성이 잠금을 소유하게 한다. `temperature=0`은 greedy이며 `do_sample=False`다. 종료는 EOS 여부/토큰 한도로 구분한다.
- [x] **Step 5:** 같은 테스트 통과 확인. 준비한 두 모델로 각 32~64 토큰의 실제 MPS 질문을 실행하고 작은 모델은 CPU에서도 확인한다. 지원하지 않는 연산·비정상 출력이 있으면 원인을 해결하고 검증 결과를 기록한다.
- [x] **Step 6:** `git add app/model_manager.py tests/test_model_manager.py` 후 `git commit -m "feat: add serialized Transformers inference"`.

## Task 3: PubMed MCP 연결과 제한된 도구 실행

**Files:** Create `app/pubmed.py`, `tests/test_pubmed.py`. Consumes Task 1.

**Interfaces:**
- `PubMedClient(url: str)`; `await client.tool_schemas() -> list[dict]`는 모델에 전달할 OpenAI 형식 도구 스키마를 반환한다.
- `await client.validate_call(name: str, arguments: dict) -> dict`: 허용 이름과 JSON Schema를 검사하고 실행 제한을 적용한 인자를 반환한다. 실제 MCP 스키마의 필드명을 사용한다.
- `await client.call(name: str, arguments: dict) -> dict`: 구조화 결과 또는 텍스트를 JSON으로 정리해 반환한다. 도구 오류는 `ServiceError`.
- `client.status() -> dict`: unknown/available/unavailable와 최근 오류; `await client.close() -> None`.
- `format_tool_result(result: dict) -> str`: 최대 12,000자이며 초과 시 JSON에 `truncated: true`와 제한한 자료를 넣는다.

- [x] **Step 1:** 테스트 작성: `test_only_two_tools_exposed`, `test_unknown_or_invalid_arguments_never_execute`, `test_search_and_fetch_limits`는 각각 5건·3건 상한을 확인한다. `test_large_result_is_bounded_and_marked`는 `len(text) <= 12000` 및 표시를 확인한다. `test_mcp_tool_error_and_timeout_are_not_success`는 SDK isError·30초 제한의 오류 변환을 확인한다. 세션 연결 수명 테스트는 도구 없는 경로에서 연결하지 않음을 확인한다.
- [x] **Step 2:** `uv run pytest tests/test_pubmed.py -q` 실행, 미구현으로 실패 확인.
- [x] **Step 3:** 공식 SDK의 `streamablehttp_client`와 `ClientSession`으로 구현한다. 각 스키마 조회·도구 실행 코루틴에서 async context로 세션을 열고 같은 코루틴에서 정리한다. 발견한 도구 스키마는 캐시하되 SDK 세션을 코루틴 간 공유하지 않는다. 도구 없는 경로에서는 연결하지 않는다. 실제 스키마로 인자를 검증하며 조회 건수 상한을 스키마에 반영하고 호출 전에도 적용한다.
- [x] **Step 4:** 테스트 통과 후 실제 기본 서버에서 `list_tools`, 검색 1회, 검색에서 얻은 PMID로 상세 조회 1회를 수행한다. 명칭·인자·PMID·결과 상태를 기록하며 가짜 PMID를 성공 기록에 넣지 않는다.
- [x] **Step 5:** `git add app/pubmed.py tests/test_pubmed.py` 후 `git commit -m "feat: connect PubMed MCP search and abstracts"`.

## Task 4: 채팅 도구 흐름과 FastAPI 공개 계약

**Files:** Create `app/schemas.py`, `app/chat_service.py`, `app/main.py`, `tests/test_chat_service.py`, `tests/test_api.py`. Consumes Tasks 1–3.

**Interfaces:**
- `ChatMessage(role: Literal['system','user','assistant'], content: str)`, `ChatRequest(model='qwen3.5-2b', messages: list[ChatMessage], max_tokens=512, temperature=0.2, use_pubmed=True)` Pydantic 모델. messages는 비어 있지 않고 숫자는 Global Constraints 범위를 적용한다.
- `ToolCall(name: str, arguments: dict)` dataclass; `parse_tool_calls(raw_text: str) -> list[ToolCall]`: Qwen `<tool_call>` 블록의 엄격한 JSON 파싱. 블록 없는 출력은 빈 목록, 불완전·잘못된 블록은 `ServiceError`.
- `ToolTrace(name: str, arguments: dict, result: dict | None, error: str | None)`; `ChatResponse`: model, choices의 assistant content 및 finish_reason, tool_calls 목록.
- `ChatService(manager: ModelManager, pubmed: PubMedClient)`; `await service.complete(request: ChatRequest) -> ChatResponse`.
- `create_app(settings: Settings | None = None, manager: ModelManager | None = None, pubmed: PubMedClient | None = None) -> FastAPI`; 모듈의 `app`은 기본 앱. 주입 가능한 객체로 HTTP 통합 테스트를 수행한다.

- [x] **Step 1:** 실패할 서비스 테스트 작성: 정상·다중·잘못된 JSON·알 수 없는 호출을 검사한다. `test_all_calls_validated_before_execution`은 정상 호출 뒤 잘못된 호출이 섞여도 외부 실행 0회를 확인한다. `test_multiple_calls_share_four_call_limit`는 한 응답의 여러 호출까지 합계 4회를 넘지 않음을 확인한다. `test_context_overflow_after_tool_results`는 재생성의 예산 오류를 반환함을 확인한다. `test_disabled_pubmed_never_connects`는 스키마 조회·도구 실행 0회와 일반 답변을 확인한다.
- [x] **Step 2:** `uv run pytest tests/test_chat_service.py -q` 실행, 미구현으로 실패 확인.
- [x] **Step 3:** 서비스와 파서를 구현한다. 매 생성의 요청된 호출 전체를 검사한 뒤 실행한다. assistant 도구 호출과 role=tool 결과는 모델 공식 템플릿이 받는 형식으로 넣는다. 결과마다 건수/크기를 제한하고 재생성한다. 최종 답변에는 EOS·제어 마커를 제거한다. 오류 details에는 이미 실행한 도구 기록을 넣고 성공 응답을 만들지 않는다. 외부 결과의 PMID·링크를 쓰도록 시스템 안내를 추가한다.
- [x] **Step 4:** HTTP 실패 테스트 작성: `test_model_list_is_exact_and_does_not_load`, `test_defaults_and_numeric_bounds`, `test_invalid_model_or_role_is_4xx`, `test_empty_messages_rejected`, `test_health_reports_unloaded_and_pubmed_unavailable`, `test_pubmed_outage_does_not_break_plain_chat`, `test_tool_failure_returns_error_and_trace`. NaN/무한 온도와 문자열 숫자는 422로 거절한다. max_tokens는 엄격한 정수, temperature는 유한한 숫자로 검증한다.
- [x] **Step 5:** `uv run pytest tests/test_api.py -q` 실행, 미구현으로 실패 확인. 이후 3개 공개 endpoint와 공통 오류 매핑/lifespan을 구현한다. 공개 모델 목록에는 Global Constraints의 2개만 있으며 지연 로드를 유지한다. 오류 상태는 입력/알 수 없는 모델 422, 문맥 초과 413, 장치·파일 불가 503, MCP·모델 도구 호출 오류 502, MCP 시간 초과 504, 예상 밖 생성 실패 500으로 고정한다.
- [x] **Step 6:** `uv run pytest tests/test_chat_service.py tests/test_api.py -q` 통과 확인. `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`로 실제 API를 시작해 `/health`, `/v1/models`, 두 모델의 도구 없는 채팅, 기본 모델의 PubMed 검색·초록 답변을 확인한다. 도구 호출 품질 문제는 템플릿·파서 문제와 구분해 해결/기록한다.
- [x] **Step 7:** 소스와 테스트를 커밋: `git commit -m "feat: expose chat API with bounded PubMed tools"`.

## Task 5: Docker와 다른 PC 실행 문서, 전체 검증

**Files:** Create `Dockerfile`, `compose.yaml`, `README.md`. Modify `.dockerignore`, `.env.example` only if execution requires exact matching settings.

**Interfaces:** `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`; 컨테이너는 host `0.0.0.0`, `DEVICE=cpu`, `/app/models` 읽기 전용 볼륨, 공개 포트 8000.

- [x] **Step 1:** CPU PyTorch를 설치하는 Python 3.12 slim Linux 이미지를 작성한다. 잠금 파일을 사용하고 dev 의존성·가중치·`.venv`는 이미지에서 제외한다. Compose가 `./models:/app/models:ro`를 연결하고 API를 worker 1개로 실행하도록 한다. 구성 파일에는 별도의 구현을 그대로 반영하는 테스트를 쓰지 않는다.
- [x] **Step 2:** README에 `uv sync --dev`, `uv run python scripts/prepare_models.py`, 위 API 실행 명령, `.venv` 활성화, 모델 선택 curl/PowerShell 예시를 작성한다. macOS/Linux의 `source .venv/bin/activate`와 Windows의 `.\.venv\Scripts\Activate.ps1`를 포함한다. NVIDIA 장비는 공식 PyTorch 선택표에서 CUDA 인덱스를 선택해 `uv pip install --reinstall torch --index-url <선택한 CUDA 인덱스>`로 교체하고 `uv run --no-sync uvicorn ...`으로 실행하도록 설명한다. 이후 `uv sync`가 잠금 파일의 CPU 배포로 복원한다는 점도 실행 안내에 포함한다. 다른 PC로 `.venv`를 복사하지 않고 다시 생성하게 한다.
- [x] **Step 3:** `uv run pytest -q`, `git diff --check`, `uv lock --check`, `docker compose config`를 실행한다. 실제 준비된 두 모델의 원본·리비전·파일 존재와 Git 추적 제외 상태도 확인한다.
- [x] **Step 4:** Docker Desktop을 실행할 수 있으면 기동 후 `docker compose build`, `docker compose up -d`를 실행한다. 상태 API와 `qwen3-0.6b`, `use_pubmed=false` CPU 채팅을 실제 확인하고 테스트 컨테이너를 정리한다. 실패하면 원인을 해결하거나 실제 제한을 README에 남기며 통과를 주장하지 않는다.
- [x] **Step 5:** README 검증 표에 실제 날짜, MPS 두 모델, CPU 작은 모델, 실제 MCP 검색/조회/도구 답변, Docker 결과를 기록한다. Windows/Linux 네이티브 및 CUDA는 현재 장비에서 미검증이라고 명시한다. 기본 모델의 실제 도구 호출 실패가 남으면 초기 구현 완료로 처리하지 않는다.
- [x] **Step 6:** 직접 실행 방식에서는 전체 변경을 새 리뷰어가 검토한다. 리뷰에서 확인된 요구 누락/결함을 수정하고 해당 위험에 필요한 검증을 다시 실행한다. 파일 책임/인터페이스 일치·모델 하나 정책·MCP 장애·취소·휴대성을 집중 확인한다.
- [x] **Step 7:** Docker/README 및 최종 수정 파일을 커밋: `git commit -m "chore: add Docker setup and portable run guide"`. 깨끗한 Git 상태와 실제 실행 명령을 최종 전달한다.

## 실행 선택

권장: **직접 실행(Native)**. 5개 작업이 모델 카탈로그와 추론·MCP 인터페이스를 순차적으로 공유하므로 같은 세션에서 구현하고 마지막에 새 리뷰어가 전체 변경을 확인하는 방식이 적합하다.

대안: **작업별 에이전트(Subagent-driven)**. 각 작업에 구현자와 리뷰어를 새로 배정하며 작업별 독립 검토 비용과 시간이 추가된다.

이 계획을 사용자에게 링크하고 검토 및 실행 방식을 확인한 뒤 구현을 시작한다.
