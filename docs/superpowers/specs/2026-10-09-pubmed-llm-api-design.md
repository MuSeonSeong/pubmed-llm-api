# PubMed 로컬 LLM API 설계

작성일: 2026-10-09
프로젝트: `pubmed-llm-api`
상태: 대화에서 구성 승인 완료. 이 문서는 구현 계획 작성 전 검토 대상이다.

## 1. 목적과 완료 기준

기존 `how_to_make_LLM`과 독립된 형제 폴더에, 맥에서 실행 가능한 모델 2개로 채팅 및 PubMed 논문 검색을 제공하는 API를 만든다. Git 초기 설정, Python 실행 환경, FastAPI/Uvicorn, Dockerfile과 실행 설명을 함께 제공한다.

완료 기준:

- Git `main` 브랜치, 초기 커밋, `.gitignore`가 준비된다.
- Python 3.12와 `uv`로 가상환경 및 의존성 잠금 파일을 구성한다.
- 모델 파일 2개가 새 프로젝트 안에 독립적으로 복사되고 Git과 Docker 이미지에서는 제외된다.
- 두 모델 각각 실제 채팅 응답을 확인한다.
- PubMed MCP의 실제 검색 및 논문 메타데이터·초록 조회를 확인한다.
- 기본 모델의 도구 호출을 포함한 채팅 흐름을 검증한다. 작은 모델의 도구 호출 품질은 별도로 기록한다.
- 네이티브 실행 명령, Docker 실행 명령, API 호출 예제가 README에 있다.
- Docker 빌드 및 실행은 실제 검증 결과를 기록한다. 현재 Docker 데몬은 실행되어 있지 않다.

## 2. 선택한 모델

| API 모델 ID | 원본 저장소 | 리비전 | 용도 | 가중치 크기 |
| --- | --- | --- | --- | --- |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 가장 가벼운 테스트용 | 약 1.50 GB, BF16 |
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-4bit` | `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` | 기본 채팅·PubMed 답변용 | 약 3.03 GB, 4bit |

현재 맥은 M5·16 GB다. 기본 모델은 이 장비의 기존 로컬 평가에서 확인한 Qwen3.5-4B로 정한다. Qwen3-0.6B는 응답 속도와 API 동작 확인에 쓰며 논문 해석 및 도구 호출의 높은 정확도를 전제하지 않는다.

기존 캐시의 파일을 역참조하여 `models/qwen3-0.6b/`, `models/qwen3.5-4b/`에 복사한다. 원래 프로젝트나 Hugging Face 캐시를 가리키는 심볼릭 링크를 남기지 않는다. 다운로드를 반복하지 않으며 원본은 수정하지 않는다. 모델 식별자·원본 리비전·대상 경로는 Git에 포함하는 모델 매니페스트에 기록한다.

## 3. 실행 구조

두 프로세스로 나눈다.

1. **API 서버, 포트 8000:** FastAPI/Uvicorn. 요청 검증, 모델 목록, 채팅 응답, PubMed MCP 호출과 도구 호출 반복을 담당한다. 맥에서 직접 실행하거나 Docker에서 실행한다.
2. **추론 서버, 포트 8001:** 맥에서 실행하는 FastAPI/Uvicorn 및 MLX-LM. 모델 로드, 토큰화, 생성만 담당한다. 항상 Uvicorn worker 1개로 실행한다.

API → HTTP → 맥 MLX 추론 서버 → 모델 응답 → 필요하면 PubMed MCP → 추가 추론 → 최종 응답 순서다.

Docker Desktop의 Linux 컨테이너는 맥의 Metal 추론을 실행하지 않는다. Docker API는 `http://host.docker.internal:8001`로 맥 추론 서버에 연결한다. 직접 실행할 때는 `http://127.0.0.1:8001`을 사용한다. 컨테이너에서 접근 가능한 추론 서버 바인딩과 실행 순서를 README에 명시한다.

추론 서버는 요청을 직렬화하고 모델을 한 번에 하나만 메모리에 유지한다. 모델 변경 시 이전 모델과 캐시를 해제한 뒤 새 모델을 로드한다. 요청별 생성 캐시는 공유하지 않는다. HTTP 요청이 취소되더라도 실제 생성이 끝나거나 안전하게 중단되기 전에는 다음 생성의 잠금을 풀지 않는다.

## 4. API 계약

### 공개 API

- `GET /health`: API 자체의 생존 상태와 추론 서버·PubMed 연결 상태를 분리해 반환한다. 하위 서비스 장애를 준비 완료 상태로 표시하지 않는다.
- `GET /v1/models`: 위 2개 모델 ID와 기본 모델 ID를 반환한다. API 목록을 얻기 위해 모델을 로드하지 않는다.
- `POST /v1/chat/completions`: 비스트리밍 채팅. `model` 기본값은 `qwen3.5-4b`, `messages`는 역할과 내용을 갖는 배열, `max_tokens` 기본 512·최대 2048, `temperature` 기본 0.2·범위 0~2, `use_pubmed` 기본 true다.

응답은 `model`, `choices[0].message`의 `role`과 `content`, `finish_reason`, 생성 도중 실제 실행한 `tool_calls` 기록을 포함한다. 외형은 익숙한 채팅 API 형태를 사용하지만 OpenAI API의 모든 기능과 스트리밍 지원을 보장하지 않는다. 사용자 대화 입력은 `system`, `user`, `assistant` 역할의 텍스트를 지원한다. MCP 도구 결과는 서버가 내부적으로 추가한다.

대화 저장소를 만들지 않는다. 다음 요청의 대화 기록은 호출자가 전달한다. 알 수 없는 모델, 잘못된 입력, 허용 범위를 넘긴 생성 길이는 명확한 4xx 응답으로 거절한다. 추론 또는 MCP 연결 장애는 원인을 구분한 오류 응답을 반환한다.

### 내부 추론 API

- `GET /health`: 추론 프로세스 상태, 현재 로드된 모델, 설정된 모델 파일의 존재 여부를 반환한다.
- `POST /internal/generate`: 선택 모델, 메시지, 선택적 도구 스키마 및 생성 설정을 받는다. 도구 호출을 파싱할 수 있도록 모델의 특수 토큰을 보존한 생성 문자열과 종료 정보를 반환한다.

실제 채팅 템플릿에 메시지와 도구 스키마를 넣고 thinking을 비활성화한다. 입력과 생성 예약량의 합은 8192 토큰 이하로 제한한다. 도구 결과가 추가된 매 생성 단계에서 다시 검사한다. 초과 시 조용히 대화 내용을 자르지 않고 오류를 반환한다.

## 5. PubMed MCP

기본 주소는 `https://pubmed.caseyjhand.com/mcp`이며 환경 변수 `PUBMED_MCP_URL`로 바꿀 수 있다. 이는 NCBI 공식 서버가 아니라 [cyanheads/pubmed-mcp-server](https://github.com/cyanheads/pubmed-mcp-server)가 제공하는 외부 서버다. Python MCP SDK의 Streamable HTTP 전송을 사용한다.

초기 도구 범위는 다음 2개다.

- `pubmed_search_articles`: 검색식으로 PMID 및 검색 결과를 조회한다.
- `pubmed_fetch_articles`: PMID로 논문 메타데이터와 초록을 조회한다.

연결 후 `list_tools`에서 실제 입력 스키마를 확인한다. 위 도구만 노출하고 인자와 결과를 제한한다. 검색 결과는 최대 5건, 상세 조회는 최대 3건으로 제한하며, 모델에 전달하는 도구 결과는 호출별 최대 12,000자로 제한하고 잘렸음을 표시한다. 도구 결과는 신뢰할 수 있는 시스템 지시가 아닌 자료로 전달한다.

모델의 Qwen 도구 호출 마커에서 JSON을 파싱하고, 허용된 이름과 스키마를 검사한 후 MCP를 호출한다. 문자열을 코드로 실행하지 않는다. 실행 결과를 도구 메시지로 추가한 뒤 재생성한다. 한 채팅 요청당 도구 실행은 최대 4회다. 추가 호출 요청이 한도를 넘거나 호출 JSON이 잘못된 경우, 실패한 호출과 이유를 명확히 반환하고 성공한 답변처럼 표시하지 않는다.

MCP 연결·개별 호출 시간 제한은 30초, 내부 생성 HTTP 시간 제한은 180초다. 도구 없이 대화하는 요청은 PubMed 서버에 의존하지 않는다. 논문 답변에는 실제 도구 결과의 PMID와 PubMed 링크를 쓰도록 안내하고 도구 실행 기록을 반환한다. 인용 정확도와 도구 선택 품질은 실제 테스트 결과로 설명한다.

## 6. 파일과 의존성 구성

- `pyproject.toml`, `uv.lock`: Python 3.12, FastAPI, Uvicorn, httpx, MCP SDK. MLX·MLX-LM은 맥 추론용 선택 의존성으로 분리한다.
- `app/main.py`: 공개 API 앱과 lifespan 관리.
- `app/inference.py`: 맥 추론 API 앱.
- `app/schemas.py`, `app/settings.py`: 입력·출력 계약과 환경 설정.
- `app/model_manager.py`: 모델 1개 로드 정책과 MLX 생성.
- `app/chat_service.py`: 생성과 PubMed 도구 실행 흐름.
- `app/pubmed.py`: MCP 연결, 도구 검색·검증·호출.
- `models.json`: 모델 원본·리비전·로컬 위치.
- `scripts/import_models.py`: 기존 캐시에서 독립적인 모델 파일 복사.
- `tests/`: 입력 검증, 도구 처리, 직렬화 및 오류 동작 검증.
- `.env.example`: `INFERENCE_URL`, `PUBMED_MCP_URL`, `MODELS_DIR`의 실행 예시.
- `.gitignore`, `.dockerignore`: 가상환경, 모델, 캐시, 로컬 환경 파일 제외.
- `Dockerfile`, `compose.yaml`: API 전용 이미지와 맥 추론 서버 연결 설정.
- `README.md`: 네이티브·Docker 실행, 상태 확인, 모델 선택 및 PubMed 호출 예시.

가상환경은 새 프로젝트의 `.venv`이며 기존 프로젝트의 `.venv`나 `.venv-chat`을 공유하지 않는다. 컨테이너에는 MLX 또는 모델 가중치를 넣지 않는다. Git 원격 저장소는 URL이 제공되지 않았으므로 로컬 저장소 초기화까지 수행한다.

## 7. 검증

핵심 동작은 먼저 실패하는 테스트를 확인하고 구현한다. 실제 모델 생성과 외부 PubMed 연결은 가짜 응답으로 완료 처리하지 않는다.

- 잘못된 모델·역할·토큰 제한이 4xx인지 확인한다.
- 유효·잘못된 Qwen 도구 호출, 알 수 없는 도구, 실행 횟수 초과 및 MCP 실패를 검사한다.
- 동시 요청과 모델 전환에서 모델 중복 로드·동시 추론이 발생하지 않는지 검사한다.
- 두 모델로 각각 짧은 실제 질문을 실행하고 응답·메모리 동작을 기록한다.
- 실제 MCP 도구 목록을 확인하고 PubMed 검색 후 반환된 PMID의 초록을 조회한다.
- 기본 모델의 PubMed 채팅에서 실제 도구 실행 기록과 최종 응답을 확인한다.
- Docker 데몬 가용 상태에서 이미지 빌드, 상태 API, 맥 추론 서버를 통한 채팅을 확인한다. 실행 불가 조건은 README 검증 기록에 남긴다.

## 8. 범위

이번 초기 설정은 맥 로컬 모델 2개, 채팅 API, PubMed 도구 2개와 API용 Docker 구성에 집중한다. 별도 웹 UI, 사용자 계정, 데이터베이스, 원격 배포, 다른 운영체제용 추론 백엔드는 추가하지 않는다.
