# PubMed LLM API

PyTorch·Transformers로 실행하는 로컬 채팅 API입니다. macOS, Windows, Linux에서 동일한 Python 소스를 사용하며, PubMed MCP를 통해 논문 검색과 초록 조회를 수행합니다.

## 포함된 모델

| API ID | 원본 | 용도 | 원본 가중치 |
| --- | --- | --- | --- |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | 가벼운 동작 테스트 | 약 1.50 GB |
| `qwen3.5-2b` | Qwen/Qwen3.5-2B | 기본 채팅·PubMed | 약 4.55 GB |

두 모델 모두 공식 원본 Safetensors이며 MLX 전용 파일이 아닙니다. 정확한 리비전은 [models.json](models.json)에 고정합니다. 모델 하나만 메모리에 유지하고, 모델을 바꾸면 이전 모델을 해제합니다. 처음 요청과 모델 전환 시 로딩 시간이 발생합니다.

## 로컬 설치와 실행

[uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한 다음 이 프로젝트 폴더에서 실행합니다. `uv`가 Python 3.12와 **프로젝트 전용 `.venv`**를 준비합니다.

```sh
cd ../pubmed-llm-api
uv sync --frozen --dev
uv run python scripts/prepare_models.py
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

기본 잠금 환경의 맥 대상은 Apple Silicon·macOS 14 이상입니다. 현재 맥에는 `.venv`와 모델 2개가 준비되어 있으므로 마지막 명령으로 바로 실행할 수 있습니다. 다른 PC에서는 설치와 모델 준비부터 실행합니다. `.venv`는 다른 PC에 복사하지 않습니다. Git 소스와 `uv.lock`으로 새 환경을 만듭니다.

가상환경을 직접 활성화해 실행하려면:

```sh
# macOS / Linux
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

```powershell
# Windows PowerShell, 프로젝트 폴더에서
uv sync --frozen --dev
uv run python scripts/prepare_models.py
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell 환경에서 활성화 스크립트가 차단되어 있으면 가상환경 활성화 없이 `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`로 실행할 수 있습니다.

- 채팅 화면: <http://127.0.0.1:8000>
- API 문서: <http://127.0.0.1:8000/docs>
- 상태: <http://127.0.0.1:8000/health>
- 모델 목록: <http://127.0.0.1:8000/v1/models>

프로젝트를 어느 경로에 두어도 모델 경로 기본값은 이 프로젝트의 `models/`입니다. 다운로드 스크립트는 Hugging Face 캐시를 활용하며 모델 파일을 역참조해 저장하므로 기존 캐시 폴더의 심볼릭 링크에 의존하지 않습니다. 완료된 모델은 다시 다운로드하지 않습니다.

### 브라우저에서 채팅하기

서버를 실행한 뒤 <http://127.0.0.1:8000>을 열면 ChatGPT와 비슷한 채팅 화면을 사용할 수 있습니다. 메시지를 입력하고 Enter 또는 보내기 버튼을 누르세요. Shift+Enter는 줄바꿈입니다.

- 답변이 생성되는 대로 실시간으로 표시됩니다. 모델 로딩, PubMed 검색·초록 조회 진행 상황도 함께 보입니다.
- 답변 중에는 보내기 버튼이 **중지** 버튼으로 바뀝니다. 중지하면 서버도 생성을 멈추고, 그때까지의 답변은 남습니다. 입력창에서 Esc를 눌러도 중지됩니다.
- 마지막 답변의 **다시 생성** 버튼으로 새 답변을 받을 수 있습니다. 이전 답변은 ‹ 1 / 3 › 버전으로 남아 넘겨볼 수 있고, 다시 생성이 실패하면 원래 답변이 그대로 남습니다. 옆의 "○○로 다시"를 누르면 다른 모델로 같은 질문에 다시 답하게 해 두 모델의 답을 비교할 수 있습니다. 질문에 마우스를 올려 **수정**하면 그 뒤의 대화를 지우고 다시 답합니다. 빈 입력창에서 ↑를 누르면 마지막 질문을 수정합니다.
- 왼쪽 목록은 날짜별로 묶이고, 검색·고정(핀)·이름 변경(더블클릭 또는 연필)·삭제를 할 수 있습니다. 고정한 대화는 맨 위에 모입니다. **새 대화**는 ⌘K 또는 Ctrl+K입니다.
- 상단에서 모델을 고르고, 입력창의 **PubMed 검색**으로 논문 검색 사용 여부를 대화마다 바꿉니다.
- 입력창의 **추론**을 켜면 모델이 답하기 전에 생각하며, 생각 과정을 접을 수 있는 패널에 실시간으로 보여줍니다. 더 신중하지만 느리고 토큰을 많이 쓰므로 최대 답변 길이를 최소 2048로 올려 요청합니다. 생각하느라 답을 쓰지 못하면 **추론 없이 다시 생성**을 누르세요.
- 오른쪽 위 **설정**에서 모든 대화에 적용할 맞춤 지시문, 창의성(temperature), 최대 답변 길이, 밝은/어두운 테마를 정합니다. 대화를 JSON으로 내보내거나 모두 삭제할 수도 있습니다.
- 답변이 길이 한도로 잘리거나 중지된 경우 **이어서 생성**을 누르면 같은 답변을 끊긴 지점부터 이어 씁니다.
- 대화가 모델의 문맥 한도를 넘으면 오래된 질문과 답변부터 빼고 답하며, 몇 개를 뺐는지 알려줍니다.
- **이미지 질문**: 클립 버튼, 끌어다 놓기, 스크린샷 붙여넣기(⌘V/Ctrl+V)로 PNG·JPEG·WebP·GIF 이미지를 첨부하고 그림에 대해 물을 수 있습니다. 이미지는 Qwen3.5 · 2B만 이해하므로 첨부하면 이 모델로 바뀝니다. 한 요청에 최근 이미지 4장까지 보내며, 브라우저 저장 공간이 부족하면 오래된 대화의 이미지부터 지웁니다.
- 텍스트 파일(.txt, .md, .csv, .json, 코드 등)과 **PDF**(논문 등)도 같은 방법으로 첨부할 수 있습니다. PDF는 서버에서 글자를 추출하며, 스캔본처럼 글자가 없는 PDF나 암호가 걸린 PDF는 읽지 못합니다. 작은 모델의 문맥 한도 때문에 텍스트 첨부는 모두 합쳐 12,000자, 그리고 문맥의 약 42%(기본 약 3,400토큰)까지이며 넘으면 앞부분만 보냅니다. 한국어는 글자당 토큰이 많아 영어보다 짧게 잘립니다.
- 답변의 스피커 버튼으로 **소리 내어 읽기**를 할 수 있습니다. 브라우저에 내장된 음성을 사용하며 코드 블록은 건너뜁니다.
- 답변마다 어떤 모델이 답했는지 표시합니다. 상단의 저장 버튼으로 현재 대화를 마크다운 파일로 내려받을 수 있습니다.
- 여러 탭에서 동시에 써도 대화가 서로 지워지지 않습니다. 다른 탭에서 답변 중인 대화는 그 답변이 끝날 때까지 읽기 전용으로 진행 상황을 보여줍니다. 모델은 한 번에 한 요청만 처리하므로 다른 요청이 진행 중이면 "기다리고 있어요"로 표시됩니다.
- PubMed를 사용한 답변 아래에 **참고한 논문** 카드(제목·저널·연도·저자, 초록 확인 여부)를 보여주며 누르면 PubMed 페이지가 열립니다.
- 표, 인용, 코드 블록(복사 버튼 포함), 링크를 서식에 맞춰 표시합니다. 검색·조회 내역도 펼쳐서 확인할 수 있습니다.
- 대화 기록과 설정은 현재 브라우저의 로컬 저장소에 저장됩니다. 브라우저 데이터를 지우면 기록도 삭제됩니다. 연결 오류가 나면 질문 아래 **다시 시도**를 누르세요.

로컬 UI는 별도 프런트엔드 설치 없이 같은 서버에서 제공되며 Docker에서도 사용할 수 있습니다.

## 설정과 장치 선택

`.env.example`을 `.env`로 복사한 뒤 수정할 수 있습니다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `DEVICE` | `auto` | `cuda` → `mps` → `cpu` 순으로 가용 장치 선택 |
| `MODELS_DIR` | `models` | 프로젝트 기준 상대 경로 또는 절대 경로 |
| `MAX_CONTEXT_TOKENS` | `8192` | 입력 템플릿·도구 스키마·첨부·이미지·생성 토큰의 합, 최대 8192. 메모리가 적은 CPU 환경에서는 4096으로 줄일 수 있습니다 |
| `PUBMED_MCP_URL` | `https://pubmed.caseyjhand.com/mcp` | Streamable HTTP MCP 주소 |

맥 MPS는 FP16, CPU는 FP32, CUDA는 BF16 지원 시 BF16/그 외 FP16을 사용합니다. 명시한 장치를 사용할 수 없으면 오류를 반환합니다. 맥에서는 원본 가중치를 CPU에서 읽은 뒤 MPS로 옮깁니다. 직접 MPS로 로드할 때 발생한 프로세스 종료를 피하기 위한 경로이며 로딩 중 일시적인 메모리 사용이 늘어납니다.

Qwen3.5-2B는 CPU FP32에서 가중치만 약 9 GB가 필요하고 추가 메모리도 사용합니다. CPU 또는 Docker에서 메모리가 부족하면 `qwen3-0.6b`로 먼저 실행하세요. 맥의 직접 실행은 MPS를 사용하고, 맥 Docker는 Linux CPU를 사용합니다.

### NVIDIA CUDA — Windows / Linux

기본 잠금 환경은 Windows/Linux에 CPU PyTorch를 설치합니다. NVIDIA GPU를 사용하려면 [공식 PyTorch 설치 선택표](https://pytorch.org/get-started/locally/)에서 장비와 드라이버에 맞는 CUDA wheel 인덱스를 선택합니다. 아래 예시의 `cu130`은 선택한 인덱스로 바꾸세요.

```sh
uv pip install --reinstall 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu130
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"
uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

이 설치는 로컬 `.venv`의 PyTorch 배포를 교체합니다. 이후 `uv sync` 또는 `--no-sync` 없는 `uv run`은 잠금 파일의 CPU 배포로 복원할 수 있으므로 CUDA 실행에서는 `--no-sync`를 사용합니다. 자세한 배포 설정은 [uv의 PyTorch 문서](https://docs.astral.sh/uv/guides/integration/pytorch/)를 참고하세요. CUDA 실장비 검증은 현재 맥에서 수행하지 않았습니다.

## 채팅 호출

PubMed 없이 가볍게 테스트합니다. 일반 채팅을 외부 MCP 연결 없이 사용하려면 `use_pubmed=false`를 지정합니다.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"2+2는 얼마야?"}],"max_tokens":64,"temperature":0,"use_pubmed":false}'
```

Windows PowerShell:

```powershell
$body = @{
  model = 'qwen3-0.6b'
  messages = @(@{ role = 'user'; content = '2+2는 얼마야?' })
  max_tokens = 64
  temperature = 0
  use_pubmed = $false
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

PubMed 논문 검색과 초록 요약:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-2b","messages":[{"role":"user","content":"PubMed에서 A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity 논문을 1건 검색하고, 검색된 PMID로 초록을 조회한 뒤 한국어 두 문장으로 요약해줘."}],"max_tokens":512,"temperature":0,"use_pubmed":true}'
```

`model` 생략 시 `qwen3.5-2b`, `max_tokens` 기본 512·범위 1~2048, `temperature` 기본 0.2·범위 0~2, `use_pubmed` 기본 true, `stream`·`truncate_history`·`thinking` 기본 false입니다. `thinking: true`이면 Qwen의 생각 모드를 켜고, 생각 내용은 답변과 분리해 `choices[0].message.reasoning_content`로 반환합니다. 생각이 `max_tokens`의 75%를 넘으면 마무리 문구와 `</think>`를 넣어 생각을 닫고 남은 길이로 답하게 합니다(budget forcing). 작은 모델이 답을 쓰지 못한 채 계속 고민하는 것을 막기 위해서입니다. 메시지는 system/user/assistant 텍스트를 지원하며 최소 한 개의 user 메시지가 필요합니다. system 메시지는 템플릿 호환을 위해 앞쪽에서 병합합니다. 대화는 서버에 저장하지 않으므로 다음 요청에 필요한 기록을 포함하세요.

응답의 `choices[0].message.content`에 답변이 있고 `tool_calls`에 실행된 도구와 실제 자료가 있습니다. `sources`에는 도구가 실제로 돌려준 논문의 `pmid`, `url`, `title`, `authors`, `journal`, `year`, `doi`, `abstract_fetched`(초록 조회 여부)를 정리해 담습니다. PubMed 답변에는 실제 결과의 PMID 링크를 덧붙입니다. 검색·조회 출처 목록은 모델의 모든 문장이 검증되었다는 뜻은 아닙니다. 작은 모델의 논문 해석과 도구 선택 정확도는 별도 평가가 필요합니다.

### 스트리밍

`"stream": true`를 지정하면 `text/event-stream`으로 진행 상황을 보냅니다. 각 이벤트는 `data: {JSON}` 한 줄입니다.

| `type` | 내용 |
| --- | --- |
| `status` | `stage`: `queued`(다른 요청 처리 대기), `loading_model`(모델 로딩), `generating`(생성 시작) |
| `tool` | PubMed 도구 `name`, `arguments`, `status`(`running`/`done`/`error`) |
| `delta` | 새로 생성된 답변 텍스트 `content`. 도구 호출 표기는 보내지 않습니다 |
| `reasoning` | `thinking` 요청에서 새로 생성된 생각 텍스트 `content` |
| `done` | 비스트리밍 응답과 같은 최종 본문. 출처 링크가 붙은 최종 답변은 이 이벤트의 내용을 사용하세요 |
| `error` | `status`(HTTP 상태에 해당), `error`(`code`, `message`, `details`) |

스트림은 항상 `done` 또는 `error` 하나로 끝납니다. 요청 검증 오류(422)는 스트림 시작 전에 일반 JSON 오류로 반환합니다. 클라이언트가 연결을 끊으면 서버는 다음 토큰에서 생성을 중단하므로, 답변 중지는 연결을 끊는 것으로 충분합니다.

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"하늘이 파란 이유를 짧게 알려줘."}],"use_pubmed":false,"stream":true}'
```

### 이미지 입력

`qwen3.5-2b`는 이미지를 이해합니다. user 메시지의 `content`를 OpenAI 형식의 배열로 보내면 됩니다. 이미지는 base64 data URL(PNG, JPEG, WebP, GIF, 8 MB 이하)만 받으며 외부 URL은 가져오지 않습니다. 요청당 최대 4장이고, 이미지마다 약 256토큰 크기로 줄여 모델에 넣습니다. `qwen3-0.6b`에 이미지를 보내면 422 `vision_unsupported`입니다. `/v1/models`의 `vision`으로 지원 여부를 알 수 있습니다.

```json
{"model": "qwen3.5-2b", "use_pubmed": false, "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
  {"type": "text", "text": "이 그림을 설명해줘."}
]}]}
```

### PDF 텍스트 추출

`POST /v1/documents/extract`에 PDF를 요청 본문 그대로(`Content-Type: application/pdf`) 보내면 `text`, `pages`, `pages_read`, `truncated`, `title`을 돌려줍니다. 20 MB·앞 60쪽·200,000자까지 처리하며 암호가 걸리거나 손상된 PDF는 422입니다.

```sh
curl http://127.0.0.1:8000/v1/documents/extract -H 'Content-Type: application/pdf' --data-binary @paper.pdf
```

### 답변 이어서 생성

`"continue_final": true`이면 마지막 assistant 메시지를 끝맺지 않고 그 뒤를 이어 생성합니다. 마지막 메시지가 assistant여야 하며, 응답에는 **새로 생성한 부분만** 담기므로 클라이언트가 기존 답변 뒤에 붙입니다. 이어 쓰는 부분 앞의 공백도 그대로 보존합니다. 답변 중간에서 도구를 호출할 수 없으므로 이 요청에서는 PubMed 도구와 `thinking`을 사용하지 않습니다.

### 긴 대화 자동 정리

`"truncate_history": true`를 지정하면 문맥 한도를 넘을 때 오류 대신 가장 오래된 질문과 그 답변부터 빼고 다시 시도합니다. system 메시지와 마지막 질문은 빼지 않습니다. 뺀 메시지 수는 응답의 `dropped_messages`에 있습니다. 마지막 질문만으로도 한도를 넘으면 기존처럼 413 오류입니다.

OpenAI API의 모든 기능을 제공하지 않습니다. Qwen3의 JSON 도구 호출과 Qwen3.5의 공식 XML 도구 호출을 모두 처리합니다. MCP 서버는 NCBI 공식 서버가 아닌 [cyanheads/pubmed-mcp-server](https://github.com/cyanheads/pubmed-mcp-server)의 외부 서버입니다.

### 제한과 오류

검색은 최대 5건, 초록 조회는 최대 3건, 요청당 도구 실행은 총 4회입니다. 초록 조회 결과는 모델에 주기 전에 소속기관·날짜 목록 같은 필드를 빼고 초록을 2,500자로 줄이며, 도구 결과는 호출당 12,000자 이하로 제한합니다. 매 생성 단계에서 문맥 한도를 검사합니다. 프롬프트가 남긴 공간이 요청한 `max_tokens`보다 적으면, `truncate_history`가 켜져 있고 뺄 수 있는 이전 대화가 있는 경우 먼저 오래된 대화를 빼서 요청한 길이를 확보합니다. 더 뺄 것이 없으면 남은 공간만큼 답변 길이를 줄이고(`finish_reason: length`), 64토큰도 남지 않을 때만 413을 반환합니다. 기본값에서는 긴 대화를 명확히 거절하며, `truncate_history`를 켠 경우에만 오래된 대화를 빼고 답합니다.

| HTTP | 원인 |
| --- | --- |
| 422 | 잘못된 메시지/숫자/이미지, 알 수 없는 모델, 이미지를 지원하지 않는 모델 |
| 413 | 입력과 생성 예약 토큰 합이 문맥 한도 초과 |
| 503 | 모델 파일 또는 지정 장치 없음 |
| 502 | MCP 오류, 잘못된 도구 호출, 실행 횟수 초과 |
| 504 | PubMed 연결/호출 30초 시간 제한 |
| 500 | 모델 로드·생성의 예상 밖 오류 |

오류 응답은 `error.code`, `error.message`, `error.details`를 포함하고, 이미 실행한 도구 기록을 보존합니다. `/health`의 PubMed 상태는 최근 연결/호출에서 확인한 상태이며 최초는 `unknown`입니다. 일반 채팅 준비 상태와 PubMed 준비 상태를 구분합니다.

## Docker — CPU 실행

먼저 호스트에서 모델을 준비하고 포트 8000의 네이티브 서버를 종료하세요.

```sh
uv run python scripts/prepare_models.py
docker compose up --build -d
curl http://127.0.0.1:8000/health
# 위 일반 채팅 예시에서 qwen3-0.6b를 선택해 CPU 테스트
docker compose logs -f api
# 종료
docker compose down
```

모델은 `./models:/app/models:ro` 볼륨으로 연결합니다. 이미지에는 원본 모델과 호스트 `.venv`를 넣지 않습니다. Docker 기본 모델도 2B이므로 메모리가 작은 Docker VM에서는 요청에 `model=qwen3-0.6b`를 지정하세요. 2B CPU 사용 시 Docker VM에도 충분한 RAM이 필요합니다. NVIDIA GPU 컨테이너 설정은 이 초기 구성에 포함하지 않습니다.

## 개발·Git

```sh
uv run pytest -q
# 채팅 상태·요청·서식 검증 (Node.js 사용)
node --test tests/chat.test.mjs
uv lock --check
git status
```

로컬 Git `main`에 초기 설정과 구현 커밋을 저장합니다. ChatGPT형 웹 UI 확장은 `feat/chatgpt-web-ui` 브랜치에 있으며 검토 후 `main`에 병합합니다. 원격 저장소는 아직 설정하지 않았습니다. `.venv/`, `models/`, `.env`, 캐시는 Git에서 제외됩니다.

## 실제 검증 기록 — 2026-10-09

장비: Apple M5, RAM 16 GB. Python 3.12.13, PyTorch 2.14.1, Transformers 5.19.0, MCP SDK 1.30.0. 아래 시간은 짧은 동작 확인이며 성능·답변 품질 벤치마크가 아닙니다.

| 검증 | 결과 |
| --- | --- |
| 원본 모델 2개 준비·외부 심볼릭 링크 제거 | 확인 |
| Qwen3-0.6B MPS, 48토큰 상한 | 응답 성공, 첫 로드 포함 약 3.37초 |
| Qwen3.5-2B MPS, 48토큰 상한 | 응답 성공, 첫 로드 포함 약 8.35초 |
| Qwen3-0.6B CPU | 응답 성공, 첫 로드 포함 약 2.03초 |
| HTTP 두 모델 일반 채팅 | 각각 `2+2=4` 응답, HTTP 200 |
| 실제 PubMed 검색·초록 조회 | 성공, PMID 31542292 |
| 2B 실제 도구 채팅 | 검색→초록 조회→한국어 요약·PMID 링크, PMID 22745249, 약 29.3초 |
| 단위·HTTP 계약 테스트 | 62개 통과, 샤드 인덱스 누락·불완전 완료 표시 복구 포함 |
| Docker Linux ARM64 CPU 이미지 빌드·기동 | 성공 |
| Docker Qwen3-0.6B CPU HTTP 채팅 | HTTP 200, `2+2=4`, 최종 수정 이미지에서 첫 로드 포함 약 9.4초 |
| SSE 스트리밍 (Qwen3-0.6B MPS) | 모델 로딩 → 생성 시작 → 토큰 단위 `delta` → `done` 순서 확인 |
| 스트리밍 중 연결 끊기 | 1800토큰 요청을 3초 후 끊고 다음 요청이 약 1초 만에 응답, 생성 중단 확인 |
| 긴 대화 (프롬프트 12,663토큰) | 기본값은 413, `truncate_history`는 메시지 12개를 빼고 HTTP 200 |
| 브라우저 UI (Chromium) | 스트리밍·중지·다시 생성·질문 수정·검색·다크 모드·모바일 화면, 2B PubMed 진행 표시 확인 |
| Docker CPU 이미지 재빌드 | 웹 UI·정적 파일 HTTP 200, Qwen3-0.6B SSE 스트리밍 `done`까지 확인 |
| PDF 첨부 | 13쪽 PDF에서 36,012자 추출, 브라우저에서 PDF 내용 질문에 수치 정확히 답변, 손상된 PDF는 안내 표시 |
| 이미지 입력 (Qwen3.5-2B MPS) | 도형 그림 2장을 각각 정확히 설명, 1200×800 이미지 포함 프롬프트 약 264토큰·약 9초, 붙여넣기·새로고침 후 썸네일 유지 |
| 이어서 생성 | 두 모델 모두 끊긴 문장 뒤를 공백까지 맞춰 이어 씀, 브라우저에서 256토큰 답변을 한 메시지로 확장 |
| 여러 탭 (Chromium) | 답변 중 연 탭은 진행 중 표시·입력 잠금·내용 따라감, 답변 탭을 닫으면 다른 탭이 몇 초 안에 정리 |
| 추론 생각 예산 | 2B가 반복 입력에 1,500토큰을 생각만 하던 요청이, 예산 강제 후 1,185토큰 안에 답까지 완료 |
| 추론 모드 | Qwen3-0.6B 약 3초, Qwen3.5-2B 약 37초 생각 후 답변, 생각 과정 분리 표시 |
| Windows/Linux 네이티브, NVIDIA CUDA | 실장비 미검증 |

Qwen3.5는 플랫폼 전용 최적화 커널 없이 PyTorch 기본 구현을 사용합니다. 최초 실행 시 커널 대체 경로 안내가 표시될 수 있으며 GPU·CPU별 속도 차이가 있습니다.
