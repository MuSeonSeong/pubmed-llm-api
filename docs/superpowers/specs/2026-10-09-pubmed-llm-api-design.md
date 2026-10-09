# PubMed 로컬 LLM API 설계 — Transformers

작성일: 2026-10-09
프로젝트: `pubmed-llm-api`
상태: 대화에서 초기 구성 승인 후, 사용자 요청으로 Transformers 및 다른 PC에서 사용할 공통 소스로 변경했다. 구현 계획 작성 전 검토 대상이다.

## 1. 목적과 완료 기준

기존 `how_to_make_LLM`과 독립된 형제 폴더에 모델 2개로 채팅 및 PubMed 논문 검색을 제공하는 API를 만든다. 추론은 PyTorch·Transformers를 사용하며 macOS, Windows, Linux에서 같은 Python 소스를 사용한다. Git, 로컬 `.venv`, FastAPI/Uvicorn, Dockerfile과 실행 설명을 함께 제공한다.

완료 기준:

- Git `main`, 초기 커밋, `.gitignore`가 준비된다.
- Python 3.12, 프로젝트 전용 `.venv`, `pyproject.toml`, `uv.lock`이 준비된다.
- Transformers용 원본 모델 2개를 프로젝트의 `models/`에 저장한다. 모델과 가상환경은 Git 및 Docker 이미지에서 제외한다.
- 두 모델 각각 실제 채팅 응답을 확인한다.
- PubMed MCP의 실제 검색 및 논문 메타데이터·초록 조회를 확인한다.
- 기본 모델의 도구 호출을 포함한 채팅 흐름을 검증한다. 작은 모델의 도구 호출 품질은 별도로 기록한다.
- 맥·Windows·Linux용 설치/활성화/실행 명령과 Docker 실행 예제가 README에 있다.
- 실제로 테스트한 장치와 아직 테스트하지 않은 플랫폼을 구분해 기록한다.

## 2. 모델 선택과 파일 준비

| API 모델 ID | 원본 저장소 | 고정 리비전 | 용도 | 원본 가중치 크기 |
| --- | --- | --- | --- | --- |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 가장 가벼운 테스트용 | 1,503,300,328 bytes, 약 1.50 GB |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 기본 채팅·PubMed 답변용 | 4,548,221,488 bytes, 약 4.55 GB |

기존 설계의 MLX Qwen3.5-4B 4bit 파일을 Transformers에서 그대로 사용하지 않는다. 공식 Qwen3.5-4B 원본은 약 9.32 GB이므로, 현재 M5·16 GB 맥의 메모리 여유를 고려해 기본 모델을 Qwen3.5-2B로 조정한다. 원본 파일 크기는 추론 중 필요한 메모리와 같지 않으며 CPU FP32 로드는 가중치 메모리가 더 커진다.

Qwen3-0.6B는 기존 Hugging Face 캐시에서 파일 내용을 역참조해 복사한다. Qwen3.5-2B는 위 고정 리비전의 공식 원본을 다운로드한다. 저장 위치는 `models/qwen3-0.6b/`, `models/qwen3.5-2b/`다. 원래 프로젝트나 사용자 캐시를 가리키는 심볼릭 링크를 남기지 않고 원본 캐시는 수정하지 않는다. 원본·리비전·로컬 위치는 Git에 포함하는 매니페스트에 기록한다.

준비 스크립트는 두 모델을 내려받거나 캐시에서 복사하며, 완료된 모델은 재다운로드하지 않는다. 다른 PC에서는 해당 스크립트로 같은 리비전을 준비할 수 있다. 소스에는 현재 맥의 절대 경로를 넣지 않는다.

## 3. 공통 실행 구조

FastAPI/Uvicorn 단일 프로세스가 API, Transformers 모델 생성, PubMed MCP 도구 호출을 담당한다. 포트는 8000이며 Uvicorn worker는 1개로 실행한다. 앞서 검토했던 맥 전용 MLX 추론 프로세스와 별도 포트 8001은 사용하지 않는다.

장치 설정 `DEVICE=auto|cuda|mps|cpu`를 제공한다. auto는 CUDA, MPS, CPU 순으로 실제 가용성을 검사한다. 명시한 장치가 없으면 시작 또는 모델 로드 시 명확한 오류를 반환한다. CUDA는 지원 시 BF16, 그렇지 않으면 FP16, MPS는 FP16, CPU는 FP32를 기본으로 한다. 플랫폼 전용 FlashAttention·MLX·bitsandbytes를 필수 의존성으로 넣지 않는다.

모델은 요청 시 로드하고 한 번에 하나만 메모리에 유지한다. 생성 요청은 직렬화한다. 모델 변경 시 이전 모델과 캐시를 해제한 후 새 모델을 로드한다. 요청별 생성 캐시는 공유하지 않는다. HTTP 요청이 취소되더라도 실제 생성이 끝나거나 안전하게 중단되기 전에는 다음 생성의 잠금을 풀지 않는다.

Qwen3-0.6B는 일반 causal LM 로더, Qwen3.5-2B는 해당 아키텍처에 맞는 공식 Transformers 로더를 사용한다. Qwen3.5는 텍스트 채팅만 제공하며 이미지·영상 입력 API는 추가하지 않는다. 채팅 템플릿과 도구 호출 마커는 각 모델의 공식 토크나이저에 맞춰 처리한다.

## 4. API 계약

- `GET /health`: API 상태, 설정/선택된 장치, 현재 로드된 모델, 모델 파일 및 PubMed 연결 상태를 구분해 반환한다. 모델이 아직 로드되지 않은 상태도 표현한다.
- `GET /v1/models`: 위 2개 모델과 기본 모델 `qwen3.5-2b`를 반환한다. 모델 목록을 얻기 위해 가중치를 로드하지 않는다.
- `POST /v1/chat/completions`: 비스트리밍 채팅. `model` 기본값은 `qwen3.5-2b`, `messages`는 역할과 내용을 갖는 배열, `max_tokens` 기본 512·최대 2048, `temperature` 기본 0.2·범위 0~2, `use_pubmed` 기본 true다.

응답은 `model`, `choices[0].message`의 `role`과 `content`, `finish_reason`, 실제 실행한 `tool_calls` 기록을 포함한다. 익숙한 채팅 API 형태를 사용하지만 OpenAI API의 모든 기능과 스트리밍을 지원한다는 의미는 아니다. 사용자가 전달하는 메시지는 `system`, `user`, `assistant` 텍스트 역할을 지원한다. MCP 도구 결과는 서버가 내부적으로 추가한다.

대화를 저장하지 않는다. 다음 요청의 대화 기록은 호출자가 전달한다. 알 수 없는 모델, 잘못된 입력, 생성 길이 또는 문맥 한도 초과는 명확한 4xx 응답으로 거절한다. 모델 로드·생성 오류와 MCP 연결 실패는 원인을 구분한 오류 응답을 반환한다.

입력과 생성 예약량의 합은 기본 4096 토큰 이하로 제한한다. `MAX_CONTEXT_TOKENS`로 조정할 수 있으며 설정 상한은 8192다. 메시지와 도구 스키마를 넣은 실제 템플릿을 기준으로 검사하고, 도구 결과가 추가된 매 생성 단계에서 다시 검사한다. 초과 시 대화 내용을 조용히 자르지 않는다. thinking을 비활성화하고 도구 파싱 전에는 도구 호출용 특수 토큰을 보존한다.

## 5. PubMed MCP

기본 주소는 `https://pubmed.caseyjhand.com/mcp`이며 `PUBMED_MCP_URL`로 바꿀 수 있다. 이는 NCBI 공식 서버가 아니라 [cyanheads/pubmed-mcp-server](https://github.com/cyanheads/pubmed-mcp-server)가 제공하는 외부 서버다. Python MCP SDK의 Streamable HTTP 전송을 사용한다.

초기 도구는 `pubmed_search_articles`와 `pubmed_fetch_articles` 2개다. 연결 후 `list_tools`에서 실제 입력 스키마를 확인하고 위 도구만 노출한다. 검색 결과는 최대 5건, 상세 조회는 최대 3건, 모델에 전달하는 결과는 호출별 최대 12,000자로 제한한다. 결과가 잘렸음을 표시하며 자료를 시스템 지시로 취급하지 않는다.

모델의 Qwen 도구 호출 마커에서 JSON을 파싱하고 허용된 이름과 스키마를 검사한 후 MCP를 호출한다. 문자열을 코드로 실행하지 않는다. 결과를 도구 메시지로 추가한 뒤 다시 생성한다. 한 채팅 요청의 도구 실행은 최대 4회다. 한도 초과, 잘못된 JSON, 도구 실패는 이유와 실행 기록을 명확히 반환한다.

MCP 연결·개별 호출의 시간 제한은 30초다. `use_pubmed=false` 요청은 PubMed 연결 없이 동작한다. 논문 답변에는 실제 도구 결과의 PMID와 PubMed 링크를 쓰도록 안내하고 도구 실행 기록을 반환한다. 인용 정확도와 도구 선택 품질은 실제 테스트 결과로 설명한다.

## 6. 가상환경, 의존성과 Docker

새 프로젝트의 `.venv`에 Python 3.12와 FastAPI, Uvicorn, httpx, MCP SDK, PyTorch, Transformers, Accelerate, Safetensors를 설치한다. 기존 프로젝트의 `.venv`와 `.venv-chat`은 공유하지 않는다. 잠금 파일로 버전을 고정한다.

기본 설치는 GPU가 없는 Windows/Linux에서도 실행할 수 있게 구성한다. NVIDIA용 PyTorch 설치는 호환되는 CUDA 배포판 선택 방법을 README에 별도로 설명한다. 소스와 모델 파일은 동일하게 사용한다. macOS용 PyTorch의 MPS 가용성은 실제 로컬 실행으로 확인한다.

README에는 다음 활성화 명령과 환경을 활성화하지 않는 `uv run` 실행을 모두 설명한다.

```sh
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Docker는 같은 API와 Transformers 추론을 함께 실행하는 Linux CPU 구성을 기본으로 한다. 모델 폴더는 읽기 전용 볼륨으로 연결하고 가중치를 이미지에 포함하지 않는다. NVIDIA GPU 컨테이너를 위한 별도 Docker 구성을 이번 필수 범위에 추가하지 않는다. 맥 Docker에서 MPS 장치를 제공한다고 가정하지 않는다.

## 7. 파일 구성

- `pyproject.toml`, `uv.lock`: 실행·개발 의존성과 플랫폼에 맞는 설치 설정.
- `app/main.py`: 공개 API 앱, lifespan과 공통 오류 처리.
- `app/schemas.py`, `app/settings.py`: 계약과 환경 설정.
- `app/model_manager.py`: 장치 선택, 모델 1개 로드 정책, Transformers 생성.
- `app/chat_service.py`: 생성 및 PubMed 도구 실행 흐름.
- `app/pubmed.py`: MCP 연결과 도구 검증·호출.
- `models.json`: 공식 원본, 고정 리비전, 상대 로컬 경로.
- `scripts/prepare_models.py`: 기존 캐시 활용 및 고정 리비전 모델 준비.
- `tests/`: 입력, 도구 처리, 장치 선택, 직렬화 및 오류 검증.
- `.env.example`: `DEVICE`, `MODELS_DIR`, `MAX_CONTEXT_TOKENS`, `PUBMED_MCP_URL`.
- `.gitignore`, `.dockerignore`: 가상환경, 모델, 캐시, 로컬 환경 파일 제외.
- `Dockerfile`, `compose.yaml`: CPU API 실행 및 모델 볼륨.
- `README.md`: 로컬 설치/활성화/실행, 다른 PC의 모델 준비, Docker, API 예시와 실제 검증 기록.

모든 파일 경로는 프로젝트 기준 또는 설정값을 사용한다. Git 원격 저장소는 URL이 제공되지 않았으므로 로컬 초기화까지 수행한다.

## 8. 검증과 범위

핵심 동작은 먼저 실패하는 테스트를 확인하고 구현한다. 실제 모델 생성과 외부 PubMed 연결을 가짜 응답으로 완료 처리하지 않는다.

- 잘못된 모델·역할·토큰 한도가 4xx인지 확인한다.
- 도구 JSON, 허용 이름, 실행 횟수, MCP 실패와 도구 결과 후 문맥 초과를 검사한다.
- 장치 자동 선택과 명시 장치 불가 처리를 검사한다.
- 동시 요청·취소·모델 전환에서 중복 로드 및 동시 생성이 없는지 검사한다.
- 현재 맥에서 두 모델의 실제 MPS 채팅과 작은 모델의 CPU 채팅을 확인한다.
- 실제 MCP 목록, 검색, 반환된 PMID의 초록 조회 및 기본 모델의 도구 채팅을 확인한다.
- Docker 데몬 실행 가능 시 이미지 빌드, 상태 API, CPU 채팅을 확인한다. 현재 데몬은 실행되어 있지 않으며 실제 검증 결과를 기록한다.
- Windows/Linux 네이티브 실행과 CUDA는 해당 장비에서 실행하지 않았다면 검증 완료로 표시하지 않는다.

이번 범위는 공통 Python 소스, 모델 2개, 로컬 환경, 채팅 API, PubMed 도구 2개 및 CPU Docker 초기 구성이다. 웹 UI, 계정, 데이터베이스, 원격 배포는 추가하지 않는다.

## 참고 자료

- [Qwen3.5 Transformers 문서](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5-2B 원본](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
- [Qwen3.5-4B 원본](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
- [Transformers 설치·캐시 문서](https://huggingface.co/docs/transformers/installation)
