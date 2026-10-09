# PubMed LLM API

로컬 모델과 채팅하고 PubMed MCP로 논문 검색과 초록 조회를 하는 API와 웹 화면입니다. Apple Silicon 맥에서는 MLX 4비트 모델을 모델마다 별도 프로세스(`mlx_vlm.server` 사이드카)로 띄워 쓰고(기본 모델은 서버가 켜질 때부터, 다른 모델은 처음 쓸 때), 그 밖의 macOS·Windows·Linux와 Docker에서는 같은 Python 소스로 PyTorch·Transformers 모델을 실행합니다.

## 포함된 모델

| API ID | 원본 | 실행 | 용도 | 가중치 |
| --- | --- | --- | --- | --- |
| `qwen3.5-9b` | mlx-community/Qwen3.5-9B-MLX-4bit | MLX (Apple Silicon) | 맥의 기본 모델, 더 정확함 | 약 5.95 GB |
| `qwen3.5-4b` | mlx-community/Qwen3.5-4B-MLX-4bit | MLX (Apple Silicon) | 더 빠름 | 약 3.03 GB |
| `qwen3.5-2b` | Qwen/Qwen3.5-2B | Transformers | 다른 OS·Docker의 기본 모델 | 약 4.55 GB |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | Transformers | 가벼운 동작 확인 | 약 1.50 GB |

Qwen3.5 모델 세 개는 이미지도 이해하고, 네 모델 모두 답하기 전에 생각(추론)할 수 있습니다. 정확한 리비전은 [models.json](models.json)에 고정합니다. 어떤 모델을 쓸지는 `MODELS`로 정하며([모델 고르기](#모델-고르기-models)), 정하지 않으면 Apple Silicon에 mlx-vlm이 설치된 경우(기본 잠금 환경) MLX 모델 두 개, 그 밖에는 Transformers 모델 두 개를 씁니다. 목록의 첫 모델이 기본 모델입니다.

MLX 모델은 기본 모델만 서버가 켜질 때 불러와 계속 메모리에 둡니다. 다른 MLX 모델은 그 모델로 처음 질문할 때 불러오고(약 6~7초) 10분 동안 질문이 없으면 다시 내려 메모리를 돌려줍니다. 16 GB 맥에서 두 모델을 함께 두면 쉬는 동안 macOS가 기본 모델을 스왑으로 내보내 첫 답변이 20초 넘게 걸렸기 때문입니다(아래 [MLX 런타임](#mlx-런타임-apple-silicon)의 "메모리", 바꾸려면 `PRELOAD_MODELS`·`MLX_IDLE_MINUTES`). 이 맥북 에어 M5(16 GB)에서 앱의 스트리밍 API로 잰 짧은 질문의 첫 토큰은 9B 약 0.20초, 4B 약 0.13초(서버를 켠 뒤 첫 질문은 0.7~0.8초)이고, 9B는 생성 초당 약 20~25토큰, 프롬프트 읽기 초당 약 700토큰입니다. Transformers 모델은 한 번에 하나만 메모리에 두고 모델을 바꾸면 이전 모델을 해제하므로 처음 요청과 모델 전환 시 로딩 시간이 생깁니다.

## 로컬 설치와 실행

[uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한 다음 이 프로젝트 폴더에서 실행합니다. `uv`가 Python 3.12와 **프로젝트 전용 `.venv`**를 준비합니다.

```sh
cd ../pubmed-llm-api
uv sync --frozen --dev
uv run python scripts/prepare_models.py
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

기본 잠금 환경의 맥 대상은 Apple Silicon·macOS 14 이상이며, 이 환경에만 mlx-vlm 0.7.6이 함께 설치됩니다. `prepare_models.py`는 이 서버가 쓸 모델(위의 기본값 또는 `MODELS`)만 받습니다. 맥에서는 MLX 모델 두 개(약 9 GB)이고, 다른 모델은 `--model <ID>`(여러 번 가능)나 `--all`로 받습니다. 다른 PC에서는 설치와 모델 준비부터 실행합니다. `.venv`는 다른 PC에 복사하지 않습니다. Git 소스와 `uv.lock`으로 새 환경을 만듭니다.

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
- API 문서: <http://127.0.0.1:8000/docs> (로컬 모드에서만 제공)
- 상태: <http://127.0.0.1:8000/health>
- 모델 목록: <http://127.0.0.1:8000/v1/models>

`AUTH_ENABLED`를 따로 정하지 않고 이렇게 직접 실행하면 로그인 없는 **로컬 모드**이며 모든 대화를 `local` 사용자 하나로 저장합니다. 대화와 설정, 첨부 이미지는 처음 사용할 때 생기는 이 프로젝트의 `data/`(`data/chat.db`, `data/files/`)에 저장되고 Git에서는 제외됩니다. 친구들이 인터넷으로 접속하게 하려면 [서버 PC로 배포하기](#서버-pc로-배포하기)를 따르세요.

프로젝트를 어느 경로에 두어도 모델 경로 기본값은 이 프로젝트의 `models/`입니다. 다운로드 스크립트는 Hugging Face 캐시를 활용하며 모델 파일을 역참조해 저장하므로 기존 캐시 폴더의 심볼릭 링크에 의존하지 않습니다. 완료된 모델은 다시 다운로드하지 않습니다.

### 브라우저에서 채팅하기

서버를 실행한 뒤 <http://127.0.0.1:8000>을 열면 ChatGPT와 비슷한 채팅 화면을 사용할 수 있습니다. 메시지를 입력하고 Enter 또는 보내기 버튼을 누르세요. Shift+Enter는 줄바꿈입니다.

- 답변이 생성되는 대로 실시간으로 표시됩니다. 모델 로딩, PubMed 검색·초록 조회 진행 상황도 함께 보입니다.
- 답변 중에는 보내기 버튼이 **중지** 버튼으로 바뀝니다. 중지하면 서버도 생성을 멈추고, 그때까지의 답변은 남습니다. 입력창에서 Esc를 눌러도 중지됩니다.
- 마지막 답변의 **다시 생성** 버튼으로 새 답변을 받을 수 있습니다. 이전 답변은 ‹ 1 / 3 › 버전으로 남아 넘겨볼 수 있고, 다시 생성이 실패하면 원래 답변이 그대로 남습니다. 옆의 "○○로 다시"를 누르면 다른 모델로 같은 질문에 다시 답하게 해 두 모델의 답을 비교할 수 있습니다. 질문에 마우스를 올려 **수정**하면 새 질문으로 다시 답하고, 이전 질문과 그 뒤의 대화는 질문 아래 ‹ 1 / 2 ›로 남아 오갈 수 있습니다. 빈 입력창에서 ↑를 누르면 마지막 질문을 수정합니다.
- 첫 답변이 끝나면 모델이 대화 제목을 짧게 지어 줍니다. 직접 이름을 바꾼 대화는 그대로 둡니다.
- 왼쪽 목록은 날짜별로 묶이고, 검색·고정(핀)·이름 변경(더블클릭 또는 연필)·삭제를 할 수 있습니다. 고정한 대화는 맨 위에 모입니다. **새 대화**는 ⌘K 또는 Ctrl+K입니다.
- 입력창 아래의 모델 버튼으로 모델을 고릅니다. 메뉴에는 서버가 쓰는 모델만 이름, 특징(더 정확함·더 빠름), 이미지 지원 여부, 준비 상태("준비 중…", 아직 불러오지 않은 모델은 "첫 질문 때 불러와요")와 함께 나옵니다. 입력창의 **PubMed 검색**으로 논문 검색 사용 여부를 대화마다 바꿉니다.
- 입력창의 **추론** 버튼으로 답하기 전에 생각할 정도를 끔(바로 답변)·가볍게·보통·깊게·최대 중에서 고릅니다. 생각 과정은 접을 수 있는 패널에 실시간으로 보여줍니다. 깊을수록 신중하지만 오래 걸리며, 9B 기준으로 생각에 쓰는 시간은 최대 약 25초·100초·3.5분·7분입니다. 생각하느라 답을 쓰지 못했거나 빨리 다시 받고 싶으면 다시 생성 메뉴의 **추론 없이 다시 생성**을 누르세요.
- 오른쪽 위 **설정**에서 모든 대화에 적용할 맞춤 지시문, 창의성(temperature, 기본값은 모델 권장값), 최대 답변 길이(기본 1024, 최대 4096, 생각에 쓰는 토큰은 따로), 밝은/어두운 테마를 정합니다. 대화를 JSON으로 내보내거나 모두 삭제할 수도 있습니다.
- 답변이 길이 한도로 잘리거나 중지된 경우 **이어서 생성**을 누르면 같은 답변을 끊긴 지점부터 이어 씁니다.
- 대화가 모델의 문맥 한도를 넘으면 오래된 질문과 답변부터 빼고 답하며, 몇 개를 뺐는지 알려줍니다.
- **이미지 질문**: 클립 버튼, 끌어다 놓기, 스크린샷 붙여넣기(⌘V/Ctrl+V)로 PNG·JPEG·WebP·GIF 이미지를 첨부하고 그림에 대해 물을 수 있습니다. 이미지는 Qwen3.5 모델만 이해하므로 다른 모델에서 첨부하면 이미지를 이해하는 모델로 바뀝니다. 한 요청에 최근 이미지 4장까지 보냅니다. 첨부한 이미지는 바로 서버의 `data/files/`에 올라가며, 계정마다 대화와 합쳐 `USER_STORAGE_MB`(기본 500 MB)까지 저장할 수 있습니다.
- 텍스트 파일(.txt, .md, .csv, .json, 코드 등)과 **PDF**(논문 등)도 같은 방법으로 첨부할 수 있습니다. PDF는 서버에서 글자를 추출하며, 스캔본처럼 글자가 없는 PDF나 암호가 걸린 PDF는 읽지 못합니다. 텍스트 첨부는 모두 합쳐 12,000자, 그리고 현재 모델 문맥(`/v1/models`의 `context_tokens`)의 약 42%까지이며 넘으면 앞부분만 보냅니다. 한국어는 글자당 토큰이 많아 영어보다 짧게 잘립니다.
- 답변의 스피커 버튼으로 **소리 내어 읽기**를 할 수 있습니다. 브라우저에 내장된 음성을 사용하며 코드 블록은 건너뜁니다.
- 답변마다 어떤 모델이 답했는지 표시합니다. 상단의 저장 버튼으로 현재 대화를 마크다운 파일로 내려받을 수 있습니다.
- 여러 탭이나 여러 기기에서 동시에 써도 대화가 서로 지워지지 않습니다. 다른 탭이나 기기에서 답변 중인 대화는 그 답변이 끝날 때까지 읽기 전용입니다. MLX 모델은 여러 요청을 함께 처리합니다. Transformers 모델은 한 번에 한 요청만 처리하므로 다른 요청이 진행 중이면 "기다리고 있어요"로 표시됩니다.
- PubMed를 사용한 답변 아래에 **참고한 논문** 카드(제목·저널·연도·저자, 초록 확인 여부)를 보여주며 누르면 PubMed 페이지가 열립니다.
- 표, 인용, 코드 블록(복사 버튼 포함), 링크를 서식에 맞춰 표시합니다. 검색·조회 내역도 펼쳐서 확인할 수 있습니다.
- 대화 기록과 설정은 서버의 `data/chat.db`에 계정별로 저장됩니다. 브라우저는 화면을 빨리 띄우기 위한 사본(캐시)만 두므로, 브라우저 데이터를 지워도 기록은 남고 같은 계정으로 로그인한 다른 기기에서도 이어서 볼 수 있습니다. 연결 오류가 나면 질문 아래 **다시 시도**를 누르세요.
- 서버 저장 이전에 이 브라우저에만 저장했던 기록은, 지금 사용자(계정 모드에서는 로그인한 계정, 로컬 모드에서는 `local`)에게 아직 서버 대화가 없을 때 한 번 가져올지 묻습니다. 가져오든 거절하든 브라우저의 예전 사본은 지우지 않으며, 거절하면 다시 묻지 않습니다. 브라우저 저장소는 주소마다 따로이므로 예전에 쓰던 브라우저에서 같은 주소(예: `http://127.0.0.1:8000`)로 열어야 합니다.
- 계정 모드(`AUTH_ENABLED=true`)에서는 먼저 로그인 화면(`/login`)이 나오고, 로그인은 `SESSION_DAYS`(기본 30일) 동안 유지됩니다.

로컬 UI는 별도 프런트엔드 설치 없이 같은 서버에서 제공되며 Docker에서도 사용할 수 있습니다.

## 설정과 장치 선택

`.env.example`을 `.env`로 복사한 뒤 수정할 수 있습니다.

| 설정 | 기본값 | 의미 |
| --- | --- | --- |
| `MODELS` | (비움) | 쓸 모델 ID를 쉼표로, 첫 번째가 기본 모델. 비우면 Apple Silicon(mlx-vlm 설치)에서는 MLX 모델 전부, 그 밖에는 Transformers 모델 전부. 없는 ID가 있으면 서버가 시작하지 않습니다 |
| `PRELOAD_MODELS` | `default` | 서버가 켜질 때 백그라운드에서 불러와 데우고 계속 메모리에 둘 MLX 모델. `default`: 기본 모델만(기본 모델이 MLX 모델일 때), `all`: 모두, `none`: 없음. 나머지는 그 모델의 첫 질문 때 불러옵니다. 예전 값 `true`는 `all`, `false`는 `none`으로 읽습니다 |
| `MLX_IDLE_MINUTES` | `10` | 미리 불러오지 않은 MLX 모델이 이 시간(분) 동안 질문을 받지 않으면 사이드카를 꺼 메모리를 돌려줌. 다음 질문 때 다시 불러옵니다(약 6~7초). `0`이면 끄지 않음. `PRELOAD_MODELS=all`이면 끌 모델이 없고, `none`이면 기본 모델도 끕니다 |
| `KV_CACHE_GB` | `0` | MLX 접두 캐시의 디스크 계층(`DATA_DIR/kv-cache/`) 용량, 모든 모델 합계 GB. `0`이면 캐시를 메모리에만 둡니다. 켜면 지운 대화·계정의 캐시도 밀려날 때까지 디스크에 남습니다(아래 "접두 캐시") |
| `MLX_CHECKPOINT_INTERVAL` | `512` | Qwen3.5의 순환(linear attention) 층이 캐시된 접두에서 이어 읽을 수 있는 체크포인트 간격(토큰) |
| `MLX_CHECKPOINT_ENTRIES` | `8` | 프롬프트마다 남기는 체크포인트 수 |
| `DEVICE` | `auto` | Transformers 모델의 장치. `cuda` → `mps` → `cpu` 순으로 가용 장치 선택 |
| `MODELS_DIR` | `models` | 프로젝트 기준 상대 경로 또는 절대 경로 |
| `MAX_CONTEXT_TOKENS` | `8192` | Transformers 모델의 문맥 한도: 입력 템플릿·도구 스키마·첨부·이미지·생각·답변 토큰의 합, 최대 8192. 메모리가 적은 CPU 환경에서는 4096으로 줄일 수 있습니다. MLX 모델의 한도는 `models.json`의 `context_tokens`(32768)입니다 |
| `PUBMED_MCP_URL` | `https://pubmed.caseyjhand.com/mcp` | Streamable HTTP MCP 주소 |
| `DATA_DIR` | `data` | 계정·대화·설정을 담는 `chat.db`(SQLite)와 첨부 이미지 `files/`를 두는 폴더. 프로젝트 기준 상대 경로 또는 절대 경로 |
| `AUTH_ENABLED` | `false` | `false`: 로그인 없는 로컬 모드(`local` 사용자 하나). `true`: 계정 모드, 로그인해야 화면과 API를 쓸 수 있음. 인터넷에 공개할 때는 반드시 `true`이며 `scripts/serve.py`가 켭니다 |
| `SESSION_DAYS` | `30` | 로그인 유지 기간(일), 1~365 |
| `USER_STORAGE_MB` | `500` | 계정마다 저장할 수 있는 대화 JSON과 첨부 파일 용량의 합(MB) |

맥 MPS는 FP16, CPU는 FP32, CUDA는 BF16 지원 시 BF16/그 외 FP16을 사용합니다. 명시한 장치를 사용할 수 없으면 오류를 반환합니다. 맥에서는 원본 가중치를 CPU에서 읽은 뒤 MPS로 옮깁니다. 직접 MPS로 로드할 때 발생한 프로세스 종료를 피하기 위한 경로이며 로딩 중 일시적인 메모리 사용이 늘어납니다.

Qwen3.5-2B는 CPU FP32에서 가중치만 약 9 GB가 필요하고 추가 메모리도 사용합니다. CPU 또는 Docker에서 메모리가 부족하면 `qwen3-0.6b`로 먼저 실행하세요. 맥에서 Transformers 모델을 쓰면 MPS를 사용하고, 맥 Docker는 Linux CPU를 사용합니다.

### 모델 고르기 (`MODELS`)

```sh
MODELS=qwen3.5-4b,qwen3.5-9b uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1   # 4B가 기본, 9B도 함께
MODELS=qwen3.5-9b uv run python scripts/prepare_models.py   # 그 모델만 준비
MODELS=qwen3.5-2b,qwen3-0.6b uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1   # 맥에서도 Transformers 모델로
```

`.env`에 `MODELS=qwen3.5-9b`처럼 적어 두어도 됩니다. `/v1/models`에는 이 모델들만 나오고 `default`가 기본 모델이며, 다른 모델을 요청하면 422 `unknown_model`입니다. MLX 모델은 Apple Silicon(mlx-vlm 설치)에서만 쓸 수 있으며, 그 밖의 환경에서 `MODELS`에 넣으면 서버가 시작하지 않습니다. MLX와 Transformers 모델을 섞어 쓸 수도 있지만 메모리를 그만큼 더 씁니다.

### MLX 런타임 (Apple Silicon)

- 모델마다 `python -m app.mlx_sidecar`(mlx-vlm 0.7.6의 `mlx_vlm.server`를 감싼 프로그램) 프로세스를 하나씩 띄웁니다. 사이드카는 TCP 포트를 열지 않고 Unix 소켓에서만 요청을 받습니다. 소켓은 앱이 실행될 때마다 새로 만드는 이 사용자 전용 폴더(권한 700, 예: `$TMPDIR/pubmed-mlx-XXXXXXXX/0.sock`, 경로가 너무 길면 `/tmp` 아래)에 있고 앱이 끝나면 폴더째 지웁니다. 그래서 다른 사용자나 웹 페이지는 모델에 닿지 못하고 포트가 겹칠 일도 없으며, 시작할 때마다 새로 만든 키(환경 변수로만 전달)가 없는 요청은 같은 사용자의 다른 프로그램이 보내도 거절합니다.
- 서버가 켜지면 `PRELOAD_MODELS`가 정한 모델(기본값은 기본 모델 하나, 여럿이면 기본 모델부터 하나씩)을 백그라운드에서 불러오고, 한 토큰짜리 답변으로 데우고 토큰을 세는 토크나이저까지 읽으면 `ready`입니다(이 맥에서 9B 8.6초, 4B 7.0초, 다시 띄울 때는 6.3초·5.6초). 다른 모델은 그 모델로 온 첫 질문이 같은 방법으로 띄웁니다. 그동안에도 화면과 `/health`는 응답합니다. 로컬 모드에서는 준비가 끝난 뒤(재시작 뒤에도) PubMed 지시문과 도구 설명을 백그라운드에서 미리 읽혀 두어 첫 PubMed 질문도 빨리 시작합니다. PubMed는 10초까지만 기다리며, 느리거나 꺼져 있어도 모델 준비나 PubMed 없는 질문은 기다리지 않습니다.
- 모델 상태(`/health`의 `inference.models`, `/v1/models`의 `state`): `missing`(모델 파일 없음, `prepare_models.py` 필요), `stopped`(아직 띄우지 않았거나 쉬어서 끔, 다음 질문 때 띄움), `starting`(준비 중), `ready`, `failed`(이유는 `error`). 준비 중인 모델에 질문하면 준비될 때까지 기다리며 "모델을 불러오는 중" 상태를 보냅니다. 시작은 앱이 맡아서 하므로, 기다리던 질문을 중지하거나 창을 닫아도 모델 시작은 끝까지 진행되고 같은 모델을 기다리는 질문들은 한 번의 시작을 함께 기다립니다.
- 긴 프롬프트를 읽는 중(첫 토큰 전)에 중지를 눌러도 0.1초 안에 연결을 끊고, 모델 서버는 그 요청을 멈춥니다.
- **쉬는 모델 끄기**: 미리 불러오지 않은 모델은 마지막 질문이 끝나고 `MLX_IDLE_MINUTES`(기본 10분)가 지나면 30초 안에 사이드카를 끄고(SIGTERM, 10초 뒤에도 남아 있으면 SIGKILL) `stopped`로 표시하며, 로그에 `stopped <모델> after N min idle`을 남깁니다. 답하는 중이거나(긴 프롬프트를 읽는 중 포함) 시작하는 중인 모델은 끄지 않습니다. 끄는 중에 온 질문은 이전 프로세스가 끝난 뒤 새로 띄워 답합니다(새 사이드카도 같은 소켓 경로를 쓰기 때문).
- 사이드카가 죽으면 다음 질문 때 다시 띄웁니다. 앱을 끄면 사이드카에 SIGTERM을 보내고 10초 뒤에도 남아 있으면 강제 종료합니다. 앱이 비정상 종료되어도 사이드카가 부모 프로세스가 사라진 것을 알아채고 스스로 끝나므로 GPU 메모리를 붙잡은 프로세스가 남지 않습니다.
- 로그는 `data/logs/mlx-<모델 ID>.log`에 쌓이고, 시작할 때 20 MB를 넘었으면 비웁니다. 요청 크기만 기록하며 대화 내용은 남기지 않습니다.
- **접두 캐시**: 앞부분이 같은 프롬프트(시스템 지시문·도구 설명·이전 대화)는 다시 읽지 않습니다. 6천 토큰 프롬프트를 다시 보낼 때 첫 토큰까지 6.3초가 0.3초로 줄었습니다. 기본값(`KV_CACHE_GB=0`)에서는 캐시를 메모리에만 두므로 서버를 끄면 사라집니다. `KV_CACHE_GB`를 켜면 디스크 계층을 `data/kv-cache/`(권한 700)에 두고 모든 모델이 합쳐 그 용량까지 씁니다. 이때 mlx-vlm은 프롬프트의 토큰 ID(토크나이저로 그대로 글로 되돌릴 수 있음)를 safetensors 메타데이터에 평문으로 저장하며, 대화나 계정을 지워도 그 캐시는 지우지 않습니다. 용량 한도에 밀려 지워질 때까지 남으므로, 바로 없애려면 서버를 멈춘 뒤 `data/kv-cache/`를 지우세요. 켰다면 `chat.db`와 같은 수준으로 보호하세요.
- **계정 사이의 캐시 분리**: 계정 모드에서는 접두 캐시를 계정마다 나눠, 응답 시간으로 다른 사람의 프롬프트를 알아낼 수 없게 합니다. 나누는 값은 계정 ID와 계정을 만든 시각으로 만듭니다(SQLite는 마지막에 만든 계정을 지우면 그 ID를 다음 계정에 다시 주기 때문). 같은 이유로 계정 모드의 사이드카는 이미지 특징을 캐시하지 않습니다(mlx-vlm은 이미지 내용으로만 캐시를 찾으므로 다른 계정이 같은 이미지를 보냈는지 시간으로 알 수 있음). 같은 이미지를 다시 물으면 시각 인코더를 다시 돌립니다.
- **한국어 답변의 한자·가나 막기**: 9B는 한국어 답에 "辗转", "合併症" 같은 한자·일본어 단어를 가끔 섞습니다(13개 중 2~3개, 샘플링과 무관). 마지막 질문에 한글이 있으면 사이드카가 그런 글자를 쓰는 토큰을 생각과 답변 모두에서 막습니다. 다만 대화의 질문 어딘가에 한자·가나가 있거나, 대화의 어느 질문이나 시스템 지시문(맞춤 지시문 포함)이 그런 글을 요청하면("일본어로 번역해 줘", "중국어 간체로", "한자 표기를 알려줘", Japanese·Chinese·kanji·hanja 등) 막지 않습니다. 그래서 "일본어로 번역해 줘" 다음의 "조금 더 자연스럽게 다듬어 줘"도 막지 않습니다. 요청은 무엇으로 쓸지를 말하는 표현입니다("일본어로", "간체로", "한자 표기·뜻", "into Japanese", "translate … Chinese"). 그저 언급하거나("중국어 논문은 빼고", "한자나 중국어를 섞지 마세요", "Do not use Chinese") 그 동사를 부정한 말("한자로 쓰지 마세요", "Do not answer in Chinese")은 요청이 아니며, "한자리"·"한자세"·"한문장"·"시간체크"처럼 글자만 같은 낱말과 첨부 파일 내용도 보지 않습니다.
- 이미지는 Transformers 경로와 같은 크기(약 256토큰)로 줄여 보냅니다. 사이드카의 기본 처리기는 한 장에 1,600만 화소(약 1만 6천 토큰)까지 그대로 읽기 때문입니다.
- **메모리**: 9B는 약 6.2 GB, 4B는 가중치 약 3.03 GB(실제 사용량은 측정 전)에 대화 길이만큼의 캐시가 더해지고, 앱 자체가 약 0.5 GB를 씁니다. 기본값에서는 기본 모델 하나만 늘 메모리에 있고 다른 모델은 쓰는 동안만 올라옵니다. 4B를 아예 쓰지 않으려면 `MODELS=qwen3.5-9b`, 8 GB 맥에서는 `MODELS=qwen3.5-4b`를 권합니다.
  - 두 모델을 함께 올려 두면 16 GB 맥에서는 쉬는 동안 macOS가 쉬는 모델의 메모리를 압축하거나 스왑으로 내보내, 그다음 첫 답변이 크게 늦습니다. 아래는 이 맥북 에어 M5(16 GB)에서 사용자의 평소 앱(VS Code, 가상 머신 등)을 띄운 채 잰 값입니다.
    - 9B와 4B를 함께 올리고 150초 쉰 뒤 첫 토큰: 9B **22.7초**, 4B 0.75초. 스왑 12.3 GB 중 11.7 GB를 쓰는 중이었고 macOS가 쉬던 9B를 스왑으로 내보냈습니다. 그 전에는 같은 이유로 78초 쉰 뒤의 4B 요청이 대기열에서 14.7초를 기다린 적도 있습니다.
    - 같은 상태에서 30초마다 두 모델에 한 토큰짜리 요청을 보내 깨워 두어도 9B 3.2초, 4B 0.48초로 여전히 느렸고, 그 요청이 다른 앱을 스왑으로 밀어냈습니다.
    - 9B만 올려 두고(4B 끔) 150초 쉰 뒤에는 두 번 모두 빨랐습니다: 0.19초, 0.54초.
  - 그래서 기본값은 기본 모델 하나만 상주시키고(`PRELOAD_MODELS=default`), 다른 모델은 처음 쓸 때 불러와(약 6~7초, 화면에 "모델을 준비하고 있어요") 10분 동안 질문이 없으면 끕니다(`MLX_IDLE_MINUTES=10`).
  - 두 모델을 늘 올려 두려면 `.env`에 `PRELOAD_MODELS=all`을 적습니다(한 번만이면 `PRELOAD_MODELS=all uv run uvicorn …`). 그러면 `MLX_IDLE_MINUTES`와 관계없이 아무것도 끄지 않습니다. 모델을 바꿀 때 기다리지 않는 대신 메모리를 4B만큼 더 쓰고, 16 GB 맥에서 다른 앱과 함께 쓰면 위처럼 쉬고 난 뒤의 첫 답변이 몇 초에서 20초 넘게 늦을 수 있습니다. 메모리에 여유가 있을 때만 쓰세요.
  - 모델을 읽기 전에 wired 메모리 한도(`mx.set_wired_limit`, 6800 MB)를 정해 보았지만 쉬고 난 뒤의 첫 토큰은 나아지지 않았습니다([docs/verification.md](docs/verification.md)).

### NVIDIA CUDA — Windows / Linux

기본 잠금 환경은 Windows/Linux에 CPU PyTorch를 설치합니다. NVIDIA GPU를 사용하려면 [공식 PyTorch 설치 선택표](https://pytorch.org/get-started/locally/)에서 장비와 드라이버에 맞는 CUDA wheel 인덱스를 선택합니다. 아래 예시의 `cu130`은 선택한 인덱스로 바꾸세요.

```sh
uv pip install --reinstall 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu130
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"
uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

이 설치는 로컬 `.venv`의 PyTorch 배포를 교체합니다. 이후 `uv sync` 또는 `--no-sync` 없는 `uv run`은 잠금 파일의 CPU 배포로 복원할 수 있으므로 CUDA 실행에서는 `--no-sync`를 사용합니다. 자세한 배포 설정은 [uv의 PyTorch 문서](https://docs.astral.sh/uv/guides/integration/pytorch/)를 참고하세요. CUDA 실장비 검증은 현재 맥에서 수행하지 않았습니다.

## 채팅 호출

PubMed 없이 가볍게 테스트합니다. 일반 채팅을 외부 MCP 연결 없이 사용하려면 `use_pubmed=false`를 지정합니다. `model`을 빼면 기본 모델이 답합니다. Transformers 환경에서 더 가볍게 시험하려면 `"model":"qwen3-0.6b"`를 넣으세요. 아래 예시는 로컬 모드 기준이며, 계정 모드에서는 [로그인 쿠키](#대화-저장-api-api)를 함께 보내야 합니다(없으면 401 `unauthorized`).

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2는 얼마야?"}],"max_tokens":64,"temperature":0,"use_pubmed":false}'
```

Windows PowerShell (Transformers 모델):

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
  -d '{"messages":[{"role":"user","content":"PubMed에서 A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity 논문을 1건 검색하고, 검색된 PMID로 초록을 조회한 뒤 한국어 두 문장으로 요약해줘."}],"max_tokens":512,"temperature":0,"use_pubmed":true}'
```

`model` 생략 시 기본 모델(`/v1/models`의 `default`), `max_tokens`(답변 길이) 기본 512·범위 1~4096, `temperature` 범위 0~2·생략 시 모델 권장값(MLX Qwen3.5: 바로 답할 때 0.7·top_p 0.8·top_k 20·presence_penalty 1.5, 생각할 때 1.0·top_p 0.95; Transformers 모델은 0.2), `use_pubmed` 기본 true, `stream`·`truncate_history` 기본 false입니다.

`reasoning_effort`로 답하기 전에 생각할 정도를 정합니다: `none`(기본, 바로 답변), `light`, `medium`, `high`, `max`. 생각에 쓸 수 있는 토큰은 각각 512, 2048, 4096, 8192이며 `max_tokens`와 별도입니다. 9B가 초당 약 20토큰을 쓰므로 생각 시간은 최대 약 25초, 100초, 3.5분, 7분입니다(작은 논리 문제에는 보통 700~1000토큰). 생각 내용은 답변과 분리해 `choices[0].message.reasoning_content`로 반환합니다. 예산을 다 쓰면 생각을 닫고 답하게 합니다(budget forcing; Transformers 모델에는 마무리 문구와 `</think>`를 넣습니다). 문맥이 모자라면 답변 길이를 먼저 지키고 남는 만큼만 생각하며, 64토큰도 남지 않으면 생각 없이 답합니다. 예전 필드 `thinking: true`는 `reasoning_effort`가 없을 때 `medium`과 같습니다. 추론을 지원하지 않는 모델(`/v1/models`의 `reasoning: false`)에서는 무시합니다. 메시지는 system/user/assistant 텍스트를 지원하며 최소 한 개의 user 메시지가 필요합니다. system 메시지는 템플릿 호환을 위해 앞쪽에서 병합합니다. 이 API는 저장된 대화를 읽지 않으므로 다음 요청에 필요한 기록을 모두 포함하세요. 웹 화면의 대화 저장은 `/api/chats`가 따로 맡습니다. 선택 필드 `chat_id`(영문·숫자·`_`·`-`, 64자 이하)를 넣어 스트리밍하면, 응답이 끝날 때까지 그 대화가 같은 계정의 다른 기기에서 답변 중(`live`)으로 보입니다. 한 계정은 답변 요청을 동시에 2개까지 보낼 수 있습니다.

응답의 `choices[0].message.content`에 답변이 있고 `tool_calls`에 실행된 도구와 실제 자료가 있습니다. `sources`에는 도구가 실제로 돌려준 논문의 `pmid`, `url`, `title`, `authors`, `journal`, `year`, `doi`, `abstract_fetched`(초록 조회 여부)를 정리해 담습니다. PubMed 답변에는 실제 결과의 PMID 링크를 덧붙입니다. 검색·조회 출처 목록은 모델의 모든 문장이 검증되었다는 뜻은 아닙니다. 작은 모델의 논문 해석과 도구 선택 정확도는 별도 평가가 필요합니다.

### 스트리밍

`"stream": true`를 지정하면 `text/event-stream`으로 진행 상황을 보냅니다. 각 이벤트는 `data: {JSON}` 한 줄입니다.

| `type` | 내용 |
| --- | --- |
| `status` | `stage`: `queued`(Transformers 모델에서 다른 요청 처리 대기), `loading_model`(모델 로딩, MLX는 모델 서버가 준비 중일 때), `generating`(생성 시작) |
| `tool` | PubMed 도구 `name`, `arguments`, `status`(`running`/`done`/`error`) |
| `delta` | 새로 생성된 답변 텍스트 `content`. 도구 호출 표기는 보내지 않습니다 |
| `reasoning` | 생각하는 요청(`reasoning_effort`)에서 새로 생성된 생각 텍스트 `content` |
| `done` | 비스트리밍 응답과 같은 최종 본문. 출처 링크가 붙은 최종 답변은 이 이벤트의 내용을 사용하세요 |
| `error` | `status`(HTTP 상태에 해당), `error`(`code`, `message`, `details`) |

스트림은 항상 `done` 또는 `error` 하나로 끝납니다. 요청 검증 오류(422)는 스트림 시작 전에 일반 JSON 오류로 반환합니다. 클라이언트가 연결을 끊으면 서버는 다음 토큰에서 생성을 중단하므로, 답변 중지는 연결을 끊는 것으로 충분합니다. 15초 동안 보낼 이벤트가 없으면 `: ping` 주석 줄을 보내 Cloudflare 같은 프록시가 조용한 연결을 끊지 않게 합니다. `data:`로 시작하지 않는 줄은 무시하세요.

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"하늘이 파란 이유를 짧게 알려줘."}],"use_pubmed":false,"stream":true}'
```

### 이미지 입력

Qwen3.5 모델(`qwen3.5-9b`, `qwen3.5-4b`, `qwen3.5-2b`)은 이미지를 이해합니다. user 메시지의 `content`를 OpenAI 형식의 배열로 보내면 됩니다. 이미지는 base64 data URL(PNG, JPEG, WebP, GIF, 8 MB 이하) 또는 `POST /api/files`로 미리 올린 내 파일의 주소(`/api/files/<id>`)만 받으며 외부 URL은 가져오지 않습니다. 다른 계정의 파일이거나 없는 파일이면 422 `invalid_image`입니다. 요청당 최대 4장이고, 이미지마다 약 256토큰 크기로 줄여 모델에 넣습니다. `qwen3-0.6b`에 이미지를 보내면 422 `vision_unsupported`입니다. `/v1/models`의 `vision`으로 지원 여부를 알 수 있습니다.

```json
{"use_pubmed": false, "messages": [{"role": "user", "content": [
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

`"continue_final": true`이면 마지막 assistant 메시지를 끝맺지 않고 그 뒤를 이어 생성합니다. 마지막 메시지가 assistant여야 하며, 응답에는 **새로 생성한 부분만** 담기므로 클라이언트가 기존 답변 뒤에 붙입니다. Transformers 모델은 이어 쓰는 부분 앞의 공백도 그대로 보존합니다. MLX 모델 서버는 assistant 메시지를 이어 쓰지 못하므로 "끊긴 곳부터 이어서 쓰라"는 지시를 새 user 메시지로 보내 나머지를 받습니다. 이때는 이어 붙는 곳의 공백이 원래와 다를 수 있습니다. 답변 중간에서 도구를 호출할 수 없으므로 이 요청에서는 PubMed 도구와 추론을 사용하지 않습니다.

### 긴 대화 자동 정리

`"truncate_history": true`를 지정하면 문맥 한도를 넘을 때 오류 대신 가장 오래된 질문과 그 답변부터 빼고 다시 시도합니다. system 메시지와 마지막 질문은 빼지 않습니다. 뺀 메시지 수는 응답의 `dropped_messages`에 있습니다. 마지막 질문만으로도 한도를 넘으면 기존처럼 413 오류입니다.

OpenAI API의 모든 기능을 제공하지 않습니다. Qwen3의 JSON 도구 호출과 Qwen3.5의 공식 XML 도구 호출을 모두 처리합니다. MCP 서버는 NCBI 공식 서버가 아닌 [cyanheads/pubmed-mcp-server](https://github.com/cyanheads/pubmed-mcp-server)의 외부 서버입니다.

### 제한과 오류

검색은 최대 5건, 초록 조회는 최대 3건, 요청당 도구 실행은 총 4회입니다. 초록 조회 결과는 모델에 주기 전에 소속기관·날짜 목록 같은 필드를 빼고 초록을 2,500자로 줄이며, 도구 결과는 호출당 12,000자 이하로 제한합니다. 매 생성 단계에서 문맥 한도(MLX 모델은 `context_tokens` 32768, Transformers 모델은 `MAX_CONTEXT_TOKENS`)를 검사합니다. MLX 모델도 앱이 모델 폴더의 토크나이저와 채팅 템플릿으로 프롬프트 토큰을 직접 세고 이미지마다 약 280토큰을 더해 검사합니다. 프롬프트가 남긴 공간이 요청한 `max_tokens`보다 적으면, `truncate_history`가 켜져 있고 뺄 수 있는 이전 대화가 있는 경우 먼저 오래된 대화를 빼서 요청한 길이를 확보합니다. 더 뺄 것이 없으면 남은 공간만큼 답변 길이를 줄이고(`finish_reason: length`), 64토큰도 남지 않을 때만 413을 반환합니다. 기본값에서는 긴 대화를 명확히 거절하며, `truncate_history`를 켠 경우에만 오래된 대화를 빼고 답합니다.

| HTTP | 원인 |
| --- | --- |
| 401 | 계정 모드에서 로그인하지 않음(`unauthorized`) |
| 422 | 잘못된 메시지/숫자/이미지, 알 수 없는 모델, 이미지를 지원하지 않는 모델 |
| 413 | 입력과 생성 예약 토큰 합이 문맥 한도 초과 |
| 429 | 한 계정의 답변 요청이 동시에 2개를 넘음(`busy`) |
| 503 | 모델 파일 또는 지정 장치 없음(`model_missing`, `device_unavailable`), MLX 모델 서버를 시작하거나 연결할 수 없음(`model_unavailable`) |
| 502 | MCP 오류, 잘못된 도구 호출, 실행 횟수 초과, MLX 모델 서버의 생성 오류(`generation_failed`) |
| 504 | PubMed 연결/호출 30초 시간 제한 |
| 500 | 모델 로드·생성의 예상 밖 오류 |

오류 응답은 `error.code`, `error.message`, `error.details`를 포함하고, 이미 실행한 도구 기록을 보존합니다. `/health`의 PubMed 상태는 최근 연결/호출에서 확인한 상태이며 최초는 `unknown`입니다. 일반 채팅 준비 상태와 PubMed 준비 상태를 구분합니다.

## 대화 저장 API (`/api`)

웹 화면이 계정·대화·설정·첨부 이미지를 다루는 API입니다. 계정 모드에서는 `POST /api/login`이 주는 `pubmed_session` 쿠키(HttpOnly, `SESSION_DAYS` 동안 유효)가 있어야 하며, 없으면 `/v1/*`를 포함해 401 `unauthorized`입니다. 로컬 모드에서는 쿠키 없이 `local` 사용자로 동작합니다.

| 요청 | 내용 |
| --- | --- |
| `GET /api/session` | 로그인 없이 호출할 수 있습니다. `auth_enabled`와 현재 사용자 `user`(`{"name"}` 또는 `null`) |
| `POST /api/login` | `{"name", "password"}`로 로그인하고 쿠키를 받습니다. 틀리면 401 `invalid_login`, 15분 안에 같은 IP나 이름으로 10번 틀리면 429 `too_many_attempts`(`Retry-After` 헤더) |
| `POST /api/logout` | 로그아웃(204) |
| `GET /api/chats?since=<seq>` | `since` 이후 바뀐 대화 `chats`, 삭제된 대화 id `deleted`, 지금 답변 중인 대화 id `live`, 다음 요청에 넘길 `seq` |
| `PUT /api/chats/{id}` | 대화 JSON 저장. 본문의 `rev`가 서버보다 높을 때만 바뀝니다. 낮으면 409 `stale_chat`(서버 사본 `chat` 포함), 삭제된 대화는 410 `chat_deleted`, 4 MB 초과는 413 `chat_too_large` |
| `DELETE /api/chats/{id}` | 삭제(204). 삭제 기록이 남아 다른 기기에서도 지워집니다 |
| `GET`·`PUT /api/settings` | 화면 설정(32 KB 이하, 마지막 저장이 이김) |
| `POST /api/files` | 이미지 원본을 본문 그대로 업로드(PNG·JPEG·WebP·GIF, 8 MB 이하) → 201 `{"id", "url", "mime", "size"}`. 다른 형식은 415, 깨진 이미지는 422 |
| `GET /api/files/{id}` | 내가 올린 파일만 내려받을 수 있습니다 |

계정의 저장 용량(`USER_STORAGE_MB`)을 넘으면 413 `storage_quota`입니다. 계정 모드에서는 `/docs`, `/redoc`, `/openapi.json`을 제공하지 않고, 로그인하지 않은 `/health` 호출에는 `{"status":"ok"}`만 돌려줍니다.

curl로 계정 모드 API를 부를 때는 먼저 로그인해 쿠키를 저장합니다. `cookies.txt`에는 로그인 토큰이 들어 있으니 다 쓰면 지우세요.

```sh
curl -c cookies.txt http://127.0.0.1:8000/api/login \
  -H 'Content-Type: application/json' -d '{"name":"민수","password":"비밀번호"}'
curl -b cookies.txt http://127.0.0.1:8000/v1/models
```

## 서버 PC로 배포하기

이 PC를 서버로 두고 친구 몇 명이 각자 계정으로 인터넷에서 접속하게 할 수 있습니다. 첫 단계는 계정 만들기입니다.

```sh
uv run python scripts/users.py add <이름>      # 1. 사람마다 계정 만들기 (비밀번호를 두 번 묻습니다)
uv run python scripts/serve.py                # 2. 계정 모드로 실행해 확인 (Ctrl+C로 종료)
bash deploy/macos/install-launch-agent.sh     # 3. 로그인하면 자동으로 켜지게 등록 (macOS)
```

지금까지 로컬 모드로 쓰던 대화는 `local` 사용자 몫이라 로그인한 계정에는 보이지 않습니다. `uv run python scripts/users.py adopt-local <이름>`으로 대화·설정·이미지를 그 계정으로 옮길 수 있습니다.

마지막 4단계로 Cloudflare Tunnel을 만들어 `https://chat.example.com` 같은 주소를 이 PC의 `http://127.0.0.1:8000`에 연결하면, 친구들은 그 주소에서 로그인해 씁니다.

`scripts/serve.py`는 계정 모드(`AUTH_ENABLED=true`)를 켜고, 로그인할 수 있는 계정이 없으면 시작하지 않으며, `127.0.0.1`에서만 접속을 받고, 서버가 켜져 있는 동안 맥이 잠들지 않게 합니다. 순서별 자세한 설명, 보안 체크리스트, 백업·업데이트·문제 해결, 그리고 저장소로 Redis 대신 SQLite를 쓰는 이유는 [docs/deploy.md](docs/deploy.md)에 있습니다.

## Docker — CPU 실행

Docker 이미지는 Linux라 MLX를 쓰지 못하고 Transformers 모델 두 개를 씁니다. 먼저 호스트에서 그 모델을 준비하고 포트 8000의 네이티브 서버를 종료하세요. 맥 호스트에서는 `prepare_models.py`의 기본값이 MLX 모델이므로 모델을 지정합니다.

```sh
uv run python scripts/prepare_models.py --model qwen3.5-2b --model qwen3-0.6b
docker compose up --build -d
curl http://127.0.0.1:8000/health
# 위 일반 채팅 예시에서 qwen3-0.6b를 선택해 CPU 테스트
docker compose logs -f api
# 종료
docker compose down
```

모델은 `./models:/app/models:ro` 볼륨으로 연결합니다. 이미지에는 원본 모델과 호스트 `.venv`를 넣지 않습니다. Docker 기본 모델도 2B이므로 메모리가 작은 Docker VM에서는 요청에 `model=qwen3-0.6b`를 지정하세요. 2B CPU 사용 시 Docker VM에도 충분한 RAM이 필요합니다. NVIDIA GPU 컨테이너 설정은 이 초기 구성에 포함하지 않습니다.

대화·계정·첨부 이미지는 `./data:/app/data` 볼륨(호스트의 `data/`)에 저장되므로 컨테이너를 지워도 남습니다. 계정 모드로 쓰려면 `AUTH_ENABLED=true docker compose up -d`로 실행하고 컨테이너 안에서 계정을 만듭니다: `docker compose exec api python scripts/users.py add <이름>`. 컨테이너가 켜져 있는 동안에는 호스트에서 같은 `data/`로 서버나 `users.py`를 실행하지 마세요. 인터넷에 공개하는 서버는 Docker 대신 `scripts/serve.py`로 실행하세요. 컨테이너 안의 서버는 프록시 헤더를 믿지 않아 로그인 시도 제한이 접속자를 IP로 구분하지 못하고, 맥에서는 GPU도 쓰지 못합니다.

## 개발·Git

```sh
uv run pytest -q
# 채팅 상태·요청·서식과 서버 동기화 검증 (Node.js 사용)
node --test tests/chat.test.mjs tests/sync.test.mjs
uv lock --check
git status
```

pytest는 실제 모델이나 사이드카 없이 돕니다. MLX 사이드카는 가짜 실행기와 `httpx.MockTransport`(그리고 모델 없이 응답만 흉내 내는 Unix 소켓 서버)로 시험하므로 다른 프로그램이 GPU를 쓰고 있어도 됩니다.

로컬 Git `main`에 초기 설정과 구현 커밋을 저장합니다. ChatGPT형 웹 UI 확장은 `feat/chatgpt-web-ui` 브랜치에 있으며 검토 후 `main`에 병합합니다. 원격 저장소는 아직 설정하지 않았습니다. `.venv/`, `models/`, `data/`(대화·계정 데이터베이스, 첨부 이미지, 접두 캐시, 서버 로그), `.env`, 캐시는 Git에서 제외됩니다.

## 실제 검증 기록 — 2026-10-09

아래는 Transformers 모델 시기의 기록입니다. MLX 런타임으로 바꾼 뒤 무엇을 어떻게 확인했는지는 [docs/verification.md](docs/verification.md)에 있습니다.

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
