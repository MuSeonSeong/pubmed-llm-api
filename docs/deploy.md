# 이 PC를 서버로 쓰기

지금 쓰는 맥을 서버로 두고, 친구 몇 명이 각자 계정으로 인터넷에서 채팅 화면에 접속하게 만드는 방법입니다. 서버를 운영해 본 적이 없어도 순서대로 따라 할 수 있게 썼습니다. 명령은 모두 터미널에서 이 프로젝트 폴더로 이동한 뒤 실행합니다.

1. 계정을 만듭니다 — `uv run python scripts/users.py add <이름>`
2. 서버 모드로 실행해 확인합니다 — `uv run python scripts/serve.py`
3. 로그인하면 서버가 저절로 켜지게 합니다 — `bash deploy/macos/install-launch-agent.sh`
4. Cloudflare Tunnel로 `https://chat.example.com` 같은 주소를 이 PC에 연결합니다.
5. 보안 체크리스트를 확인하고, 백업 방법을 익혀 둡니다.

## 전체 구조

```text
친구의 브라우저  https://chat.example.com
      │
      ▼
Cloudflare       HTTPS 인증서, 공격 차단
      │          이 PC가 바깥으로 먼저 열어 둔 암호화 터널 (공유기 포트는 열지 않습니다)
      ▼
cloudflared      이 PC에서 실행되는 터널 프로그램
      │          http://127.0.0.1:8000 (이 PC 안에서만 오가는 연결)
      ▼
앱               scripts/serve.py가 실행하는 FastAPI 서버와 Qwen 모델 (프로세스 1개)
      │
      ▼
data/chat.db     SQLite: 계정, 로그인 세션, 대화, 설정
data/files/      첨부한 이미지
```

- 앱은 `127.0.0.1`, 즉 이 PC 자신에게서 오는 접속만 받습니다. 바깥 요청은 모두 Cloudflare와 cloudflared를 거쳐 들어오므로 공유기 설정을 바꿀 필요가 없습니다.
- 대화의 원본은 서버의 `data/chat.db`에 계정별로 저장됩니다. 브라우저는 화면을 빨리 띄우려고 사본(캐시)만 가지고 있으므로, 다른 기기에서 같은 계정으로 로그인해도 같은 대화가 보입니다.
- `data/`는 `.gitignore`에 들어 있어 Git에 올라가지 않습니다. 비밀번호 해시와 대화가 들어 있으니 다른 곳에 올리지 마세요.

## 저장소로 SQLite를 쓰는 이유 (Redis·PostgreSQL 대신)

대화 저장소로 Redis를 먼저 떠올리기 쉽지만, 이 서비스에는 SQLite가 더 잘 맞습니다. 판단 기준은 이 서비스가 실제로 하는 일입니다.

| 이 서비스의 특징 | 그래서 저장소에 필요한 것 |
| --- | --- |
| 사용자는 친구 몇 명이고, 계정마다 동시에 받는 답변은 2개까지 | 많은 동시 접속을 견디는 성능은 필요 없음 |
| 서버 프로세스는 하나 (모델은 PC 한 대에 한 번만 올릴 수 있음) | 여러 서버가 상태를 나눠 가질 일이 없음 |
| 답변이 나오는 동안 약 1.5초마다 그 대화를 저장 | 초당 몇 건 정도의 작은 쓰기 |
| 대화 하나는 수 KB~수 MB의 JSON, 이미지는 따로 파일 | 큰 문서를 디스크에 오래 보관 |
| 재시작하거나 전원이 꺼져도 기록이 남아야 함 | 쓰기가 바로 디스크에 기록됨 |
| 백업이 쉬워야 함 | 파일 몇 개로 끝나는 구조 |
| 16 GB 맥에서 모델과 RAM을 나눠 씀 | 데이터를 RAM에 올려 두지 않음 |

**SQLite**는 파이썬에 기본으로 들어 있는 데이터베이스입니다. 따로 설치하거나 켜 둘 프로그램 없이 `data/chat.db` 파일 하나에 모든 것을 저장합니다. WAL 모드라 읽기와 쓰기가 서로 기다리지 않고, 갑자기 전원이 꺼져도 파일이 깨지지 않도록 설계되어 있습니다. 이 서비스의 쓰기(초당 몇 건)는 SQLite가 감당하는 양에 비하면 아주 적습니다. 백업도 명령 한 줄입니다.

**Redis**는 데이터를 전부 메모리(RAM)에 두는 저장소입니다. 아주 빠르지만 이 서비스에서는 단점이 더 큽니다.

- 대화와 이미지를 모두 RAM에 올려 두어야 하므로 모델이 써야 할 16 GB를 함께 나눠 씁니다.
- 기본 설정은 변경량에 따라 몇 분에서 한 시간 간격으로 한꺼번에 디스크에 저장(스냅샷)하므로, 갑자기 꺼지면 최근 기록이 사라질 수 있습니다. 쓰기마다 기록을 남기는 AOF를 켜도 기본값(`everysec`)에서는 마지막 1초 분량을 잃을 수 있습니다.
- 별도 서버 프로그램을 설치하고, 항상 켜 두고, 업데이트하고, 비밀번호를 걸어 관리해야 합니다.

**PostgreSQL**은 훌륭한 데이터베이스지만 역시 별도 서버 프로그램(설치, 실행, 업데이트, 계정 관리, 백업 도구)이 필요합니다. 지금 규모에서는 관리할 것만 늘어납니다.

**Redis가 맞는 때**는 사용자가 늘어 앱 서버를 여러 대(또는 여러 프로세스)로 나눌 때입니다. 그러면 로그인 세션, '답변 중' 표시, 로그인 시도 횟수처럼 잠깐 쓰고 버리는 상태를 서버끼리 공유해야 합니다. 모델을 별도 GPU PC로 옮겨 요청을 줄 세우는 작업 큐가 필요할 때나 자주 읽는 결과를 캐시할 때도 Redis가 좋습니다. 그때도 대화 원본은 SQLite나 PostgreSQL에 두고 Redis는 캐시·큐로만 쓰는 것이 일반적입니다. 여러 PC의 앱 서버가 같은 대화 기록을 함께 써야 하면 그때 PostgreSQL로 옮기면 됩니다.

## 0. 준비

README의 설치 순서대로 환경과 모델을 준비합니다.

```sh
cd ~/workspace/any/pubmed-llm-api     # 프로젝트를 둔 실제 경로
uv sync --frozen --dev
uv run python scripts/prepare_models.py
```

- 서버 PC는 전원에 연결해 두고, 맥북이라면 덮개를 연 채로 둡니다.
- 프로젝트 폴더를 `문서`, `데스크톱`, `다운로드` 폴더 안에 두지 마세요. macOS 개인정보 보호 때문에 자동으로 켜진 서버가 그 폴더의 파일을 읽지 못하거나 허용을 묻는 창이 뜰 수 있습니다.
- 맥에서는 Docker 대신 이 문서처럼 직접 실행하세요. 맥의 Docker는 Linux CPU로 돌아 GPU(MPS)를 쓰지 못하므로 훨씬 느립니다.

## 1. 계정 만들기

앱에는 두 가지 모드가 있습니다.

| 모드 | 설정 | 쓰는 때 |
| --- | --- | --- |
| 로컬 모드 | `AUTH_ENABLED=false` (기본값) | 이 PC에서 혼자 쓸 때. 로그인 없이 `local`이라는 사용자 하나로 저장합니다 |
| 계정 모드 | `AUTH_ENABLED=true` | 다른 사람이 접속할 때. 로그인해야 쓸 수 있고 대화는 계정마다 따로 보관됩니다. `scripts/serve.py`가 자동으로 켭니다 |

계정은 사람마다 하나씩 만듭니다.

```sh
uv run python scripts/users.py add 민수            # 비밀번호를 두 번 묻습니다 (입력한 글자는 화면에 보이지 않습니다)
uv run python scripts/users.py list                # 계정 목록
uv run python scripts/users.py passwd 민수         # 비밀번호 바꾸기
uv run python scripts/users.py remove 민수 --yes   # 계정과 그 계정의 데이터 삭제 (되돌릴 수 없습니다)
uv run python scripts/users.py adopt-local 민수    # 로컬 모드에서 쓰던 대화·설정·이미지를 이 계정으로 옮기기
```

- 이름은 1~32자의 영문·숫자·`_`·`.`·`-`·한글이고 `local`은 쓸 수 없습니다. 비밀번호는 8자 이상이어야 하며 12자 이상을 권합니다.
- 본인 계정도 만드세요. 계정 모드에서는 서버 PC에서도 로그인해야 합니다.
- 비밀번호는 원래 글자가 아니라 scrypt 해시로만 저장되므로 관리자도 알 수 없습니다. 비밀번호를 잊은 사람에게는 `passwd`로 새 비밀번호를 정해 주세요.
- 서버가 켜져 있어도 계정을 추가하거나 바꿀 수 있습니다.
- 다른 스크립트에서 계정을 만들 때는 `--password-stdin`을 붙여 표준 입력으로 비밀번호를 넘깁니다. 셸 기록에 비밀번호가 남지 않게 주의하세요.

### 예전 대화 기록

- 서버 저장이 생기기 전에 브라우저에만 저장했던 대화는, 로그인한 계정에 아직 서버 대화가 하나도 없으면 가져올지 한 번 묻습니다. 가져오면 그 계정의 서버 대화가 되고, 거절하면 다시 묻지 않습니다. 어느 쪽이든 브라우저의 예전 사본은 지우지 않습니다.
- 브라우저 저장소는 주소마다 따로입니다. 예전에 쓰던 그 브라우저에서 예전과 같은 주소(예: `http://127.0.0.1:8000`)로 열어야 이 안내가 나옵니다. `https://chat.example.com`으로 열면 예전 기록이 보이지 않습니다.
- 로컬 모드에서 저장한 대화(로컬 모드에서 가져온 예전 기록 포함)는 `local` 사용자 몫이라 계정 모드에서는 보이지 않습니다. 지워지지는 않으며, `uv run python scripts/users.py adopt-local <이름>`으로 대화·설정·이미지를 그 계정으로 옮길 수 있습니다. 같은 대화가 이미 계정에 있으면 계정의 것을 그대로 두고, 설정은 계정에 아직 없을 때만 옮깁니다.

## 2. 서버 모드로 실행하기

```sh
uv run python scripts/serve.py
```

`scripts/serve.py`는 인터넷 공개에 필요한 설정을 한 번에 켜는 실행기입니다.

1. 계정 모드(`AUTH_ENABLED=true`)로 실행합니다. `.env`에 `AUTH_ENABLED=false`가 있어도 계정 모드가 됩니다. 터미널에서 `AUTH_ENABLED=false`를 직접 지정하면 실행을 거부합니다.
2. 로그인할 수 있는 계정이 하나도 없으면 실행하지 않고 계정 만드는 방법을 알려 줍니다.
3. `127.0.0.1:8000`에서만 접속을 받습니다. 인터넷에서 오는 요청은 cloudflared가 이 주소로 전달합니다.
4. 서버가 켜져 있는 동안 맥이 저절로 잠들지 않게 합니다(`caffeinate -i`).
5. 서버 프로세스(워커)는 하나만 띄웁니다. 프로세스마다 모델을 따로 메모리에 올리기 때문입니다.
6. `127.0.0.1`에서 온 `X-Forwarded-For`·`X-Forwarded-Proto` 헤더만 믿습니다. 그래서 로그인 시도 제한이 실제 접속자 IP를 기준으로 동작하고, https로 접속하면 로그인 쿠키에 `Secure`가 붙습니다.

브라우저로 <http://127.0.0.1:8000>을 열면 로그인 화면이 나옵니다. 만든 계정으로 로그인해 보세요. 끝내려면 터미널에서 Ctrl+C를 누릅니다.

| 옵션 | 뜻 |
| --- | --- |
| `--port 8001` | 다른 포트를 씁니다(기본 8000). Cloudflare 쪽 주소의 포트도 같이 바꾸세요 |
| `--allow-lan` | `0.0.0.0`에 열어 같은 와이파이의 다른 기기가 `http://<이 PC의 IP>:8000`으로 직접 접속하게 합니다. 암호화되지 않으므로 잠깐 시험할 때만 쓰세요 |
| `--no-caffeinate` | 잠자기 방지를 쓰지 않습니다 |
| `--check` | 서버를 켜지 않고 시작할 수 있는지만 확인합니다 |

이미 `uv run uvicorn ...`으로 켜 둔 서버가 있으면 먼저 끄세요. 두 서버가 같은 포트를 쓸 수 없습니다.

## 3. 로그인하면 자동으로 켜지게 하기 (macOS)

터미널을 닫으면 서버도 꺼집니다. 맥에 로그인할 때마다 서버가 저절로 켜지게 하려면 LaunchAgent로 등록합니다.

**LaunchAgent란?** macOS에는 프로그램을 대신 켜고 꺼 주는 launchd라는 관리자가 있습니다. LaunchAgent는 launchd에게 "이 사용자가 로그인하면 이 프로그램을 켜고, 꺼지면 다시 켜 줘"라고 적어 두는 작은 설정 파일(plist)입니다. 이 프로젝트의 설정 파일은 `~/Library/LaunchAgents/com.pubmedchat.server.plist`에 만들어집니다.

```sh
bash deploy/macos/install-launch-agent.sh --dry-run   # 만들 설정 파일을 보여 주기만 하고 아무것도 바꾸지 않습니다
bash deploy/macos/install-launch-agent.sh             # 등록하고 바로 시작합니다
```

설치 스크립트는 먼저 계정이 있는지 확인하고, 없으면 등록하지 않습니다. 그다음 로그인할 때 이 프로젝트 폴더에서 `uv run python scripts/serve.py`를 실행하라는 설정 파일을 만들고(launchd는 PATH가 짧아 uv를 절대 경로로 적습니다), `launchctl bootstrap`으로 등록한 뒤 `launchctl kickstart`로 바로 시작합니다. 서버 로그는 `data/logs/`에 쌓입니다. sudo 없이 실행하세요. 등록하면 macOS가 "백그라운드 항목이 추가됨" 알림을 띄울 수 있습니다. 시스템 설정 › 일반 › 로그인 항목(및 확장 프로그램)에서 이 항목(이름이 uv로 보일 수 있습니다)의 백그라운드 실행을 끄면 서버가 켜지지 않습니다.

**이렇게 동작합니다**

- **로그인한 뒤에 켜집니다.** 맥의 전원만 켜고 로그인하지 않으면 서버도 꺼져 있습니다. 화면을 잠가도 되지만 로그아웃하면 서버도 꺼집니다.
- 정전이나 macOS 업데이트로 재시동되면 다시 로그인해야 켜집니다. 시스템 설정 › 사용자 및 그룹에서 자동 로그인을 켜면 로그인 없이 다시 켜지지만, FileVault(디스크 암호화)를 켠 맥에서는 자동 로그인을 쓸 수 없습니다.
- **꺼지면 다시 켭니다.** 서버가 오류로 종료되면 launchd가 다시 켭니다. 너무 자주 반복하지 않도록 최소 30초 간격을 둡니다.
- **잠자기**: 서버가 켜져 있는 동안 `caffeinate -i`가 맥이 저절로 잠들지 않게 막습니다. 화면은 평소처럼 꺼져도 됩니다. 하지만 맥북 덮개를 닫으면 잠듭니다(전원과 외부 모니터를 연결한 상태는 예외). 맥북은 덮개를 연 채 전원에 연결해 두세요.
- 자동 시작이 켜져 있는 동안에는 터미널에서 서버를 또 실행하지 마세요(포트가 겹칩니다). 잠깐 직접 실행하고 싶으면 아래 명령으로 먼저 멈춥니다.
- 프로젝트 폴더를 옮기거나 uv를 다시 설치했다면 설치 스크립트를 다시 실행하세요. 설정 파일에 절대 경로가 들어 있습니다.

**상태와 로그 확인**

```sh
launchctl print gui/$(id -u)/com.pubmedchat.server | grep -E 'state =|pid =|last exit code'
curl http://127.0.0.1:8000/health      # {"status":"ok"}가 나오면 정상
tail -n 50 data/logs/server.log        # 서버 시작·종료·오류 메시지
tail -f data/logs/access.log           # 접속 기록을 실시간으로 (그만 보려면 Ctrl+C)
pmset -g assertions | grep caffeinate  # 잠자기 방지가 걸려 있는지
```

`state = running`이면 실행 중입니다. 로그 파일은 계속 커지므로 가끔 `du -h data/logs`로 크기를 확인하고, 크면 `: > data/logs/access.log`로 비우세요. 서버가 켜져 있어도 비울 수 있습니다.

**다시 시작, 멈추기, 해제**

```sh
launchctl kickstart -k gui/$(id -u)/com.pubmedchat.server   # 다시 시작 (업데이트 후)
launchctl bootout gui/$(id -u)/com.pubmedchat.server        # 지금 멈추기 (다음 로그인 때 다시 켜집니다)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pubmedchat.server.plist   # 멈춘 서버 다시 켜기
bash deploy/macos/uninstall-launch-agent.sh                 # 자동 시작 해제 (대화와 계정은 그대로 남습니다)
```

## 4. 인터넷에서 접속하게 하기

| 방법 | 주소 | 비용 | 쓰는 때 |
| --- | --- | --- | --- |
| A. Cloudflare Tunnel + 내 도메인 | `https://chat.example.com` | 도메인 값 (터널은 무료) | 계속 쓸 때 (권장) |
| B. Cloudflare Quick Tunnel | `https://<무작위>.trycloudflare.com` | 무료 | 잠깐 시험할 때만 |
| C. Tailscale Funnel | `https://<기기>.<tailnet>.ts.net` | 무료 | 도메인 없이 계속 쓸 때 |

어느 방법이든 앱은 계정 모드(`scripts/serve.py`)로 실행해야 합니다.

### A. Cloudflare Tunnel과 내 도메인 (권장)

Cloudflare Tunnel은 이 PC에서 Cloudflare 쪽으로 암호화된 연결을 먼저 열어 두고, 그 연결로 방문자의 요청을 받아 오는 방식입니다. 공유기 포트를 열지 않아도 되고, HTTPS 인증서와 공격 차단은 Cloudflare가 맡습니다. Cloudflare 계정(무료)과 Cloudflare에 연결된 도메인이 필요합니다. 아래에서는 `chat.example.com`을 예로 듭니다.

**1) 도메인 준비**

- 새로 사려면 Cloudflare 대시보드의 Domain Registration › Register Domains에서 원하는 이름을 검색해 삽니다. Cloudflare Registrar는 등록 원가로 팔고, 산 도메인은 처음부터 Cloudflare에 연결됩니다.
- 이미 다른 곳(가비아, 후이즈 등)에서 산 도메인이 있다면 Cloudflare 대시보드에서 그 도메인을 추가하고(Free 요금제), 안내되는 네임서버 2개를 원래 등록 업체의 네임서버 설정에 넣습니다. 적용에는 보통 몇 분에서 몇 시간, 길게는 하루가 걸리며 끝나면 Cloudflare가 메일로 알려 줍니다.

**2) cloudflared 설치**

```sh
brew install cloudflared      # Homebrew가 없다면 https://brew.sh 에서 먼저 설치합니다
```

**3) 대시보드에서 터널 만들기 (가장 쉬운 방법)**

1. Cloudflare 대시보드에서 Networking › Tunnels(예전 화면은 Zero Trust › Networks › Tunnels)로 가서 **Create a tunnel**을 누르고, 종류는 Cloudflared를 고릅니다.
2. 터널 이름(예: `pubmed-chat`)을 정하면 운영체제별 설치 명령이 나옵니다. macOS를 고르고, 긴 토큰이 붙은 `cloudflared service install` 명령을 복사해 이 맥의 터미널에서 실행합니다.

   ```sh
   sudo cloudflared service install <토큰>
   ```

   cloudflared가 시스템 서비스로 등록되어 맥이 켜지면 저절로 시작합니다. 토큰은 터널의 비밀번호와 같으니 남에게 보여 주지 마세요.
3. 연결됨(Connected)이 보이면 다음으로 넘어가 경로를 추가합니다. 화면에 따라 **Routes › Add route › Published application** 또는 예전 이름인 **Public Hostname**입니다.
   - Subdomain `chat`, Domain `example.com`
   - Service URL `http://127.0.0.1:8000` (예전 화면은 Type `HTTP`, URL `127.0.0.1:8000`). Cloudflare 예시처럼 `localhost:8000`이라고 적어도 동작하지만, 앱이 IPv4 주소 127.0.0.1에서만 받으므로 이렇게 적는 편이 확실합니다.
4. `https://chat.example.com`을 열어 로그인 화면이 나오면 끝입니다. DNS 기록은 Cloudflare가 자동으로 만듭니다.

cloudflared는 맥이 켜지면 바로 시작하지만 앱은 로그인한 뒤에 켜지므로, 로그인 전에는 502 오류가 보입니다.

**직접 관리하는 방식 (설정 파일)**

대시보드 대신 설정 파일로 관리하고 싶다면 [deploy/cloudflared/config.example.yml](../deploy/cloudflared/config.example.yml)을 씁니다.

```sh
cloudflared tunnel login                                    # 브라우저가 열리면 도메인을 고릅니다 (~/.cloudflared/cert.pem 생성)
cloudflared tunnel create pubmed-chat                       # 터널 ID가 나오고 ~/.cloudflared/<터널 ID>.json이 생깁니다
cloudflared tunnel route dns pubmed-chat chat.example.com   # 주소를 이 터널에 연결합니다
cp deploy/cloudflared/config.example.yml ~/.cloudflared/config.yml
open -e ~/.cloudflared/config.yml                           # <TUNNEL-ID>, <사용자 이름>, 주소를 바꿔 저장합니다
cloudflared tunnel ingress validate                         # OK가 나오면 설정이 올바릅니다
cloudflared tunnel run pubmed-chat                          # 실행해 보고, 접속을 확인했으면 Ctrl+C
cloudflared service install                                 # 로그인하면 저절로 켜지게 등록합니다
```

sudo 없이 `cloudflared service install`을 실행하면 로그인할 때 켜지는 LaunchAgent로 등록되고 `~/.cloudflared/`의 설정을 씁니다. 앱도 로그인할 때 켜지므로 이 방식과 잘 맞습니다. sudo로 실행하면 부팅할 때 켜지는 대신 설정과 인증 파일을 `/etc/cloudflared/`에 두어야 합니다.

### B. 도메인 없이 잠깐 시험하기 (Quick Tunnel)

```sh
cloudflared tunnel --url http://127.0.0.1:8000
```

- 출력에 `https://<무작위 단어>.trycloudflare.com` 주소가 나옵니다. Cloudflare 계정도 도메인도 필요 없습니다.
- 명령을 실행하는 동안만 열리고 실행할 때마다 주소가 바뀝니다. 가동을 보장하지 않는 시험용입니다.
- **답변이 실시간으로 나오지 않을 수 있습니다.** Cloudflare 문서에 따르면 Quick Tunnel은 SSE(Server-Sent Events)를 지원하지 않는데, 이 앱은 답변을 SSE로 보냅니다. 로그인과 화면 확인용으로만 쓰세요.
- 주소를 아는 사람은 누구나 로그인 화면까지 볼 수 있으므로 앱은 반드시 계정 모드로 실행합니다.
- `~/.cloudflared/config.yml`이 있으면 Quick Tunnel이 동작하지 않을 수 있습니다. 잠시 이름을 바꿔 두세요.

### C. Tailscale Funnel

[Tailscale](https://tailscale.com)은 내 기기들을 하나의 사설 네트워크(tailnet)로 묶어 주는 서비스이고, Funnel은 그중 한 포트를 인터넷에 공개하는 기능입니다. 도메인 없이 `https://<기기 이름>.<tailnet 이름>.ts.net` 주소가 생깁니다.

```sh
tailscale funnel --bg 8000    # 처음에는 관리 페이지에서 Funnel 사용을 허용하라는 안내가 나옵니다
tailscale funnel status       # 공개 주소 확인
tailscale funnel reset        # 공개 중지
```

- `--bg`로 켜 두면 재시동한 뒤에도 다시 공개됩니다. 무료지만 베타 기능이고 대역폭 제한이 있습니다.
- 친구들도 Tailscale을 설치할 수 있다면, 인터넷 전체에 공개하는 Funnel 대신 `tailscale serve --bg 8000`으로 tailnet 안에만 열고 친구를 초대하는 편이 더 안전합니다.

### 공유기 포트포워딩은 권하지 않습니다

- 집 IP 주소가 인터넷 전체에 드러나고, 자동 스캐너가 곧바로 접속을 시도합니다.
- HTTPS 인증서를 직접 발급하고 갱신해야 합니다. 없으면 비밀번호가 암호화되지 않은 채 오갑니다.
- 가정용 인터넷은 IP가 바뀌기도 하고, 통신사가 들어오는 포트를 막기도 하며, 공유기마다 설정 방법이 다릅니다.
- 앱을 `127.0.0.1` 대신 `0.0.0.0`에 열어야 하므로(`--allow-lan`) 같은 네트워크의 모든 기기에도 그대로 노출됩니다.

Cloudflare Tunnel과 Tailscale은 이 PC에서 바깥으로 나가는 연결만 쓰므로 이런 문제가 없습니다.

## 5. 보안 체크리스트

- [ ] 인터넷에 공개하는 서버는 항상 `scripts/serve.py`(계정 모드)로 실행합니다. **로컬 모드(`AUTH_ENABLED=false`) 서버를 터널 뒤에 두지 마세요.** 주소를 아는 누구나 로그인 없이 모델을 쓰고 `local` 사용자의 대화를 볼 수 있습니다.
- [ ] 친구마다 계정을 따로 만들고, 계정을 함께 쓰지 않습니다.
- [ ] 비밀번호는 12자 이상, 다른 사이트에서 쓰지 않는 것으로 정합니다. 비밀번호 관리자가 만들어 주는 무작위 비밀번호가 좋습니다.
- [ ] 앱은 `127.0.0.1`에만 엽니다(기본값). 공개 서버에서는 `--allow-lan`과 공유기 포트포워딩을 쓰지 않습니다.
- [ ] 터널 토큰과 `~/.cloudflared/`의 인증 파일은 비밀번호처럼 다룹니다.
- [ ] 쓰지 않는 계정은 `users.py remove`로 지우고, 비밀번호가 새었다고 의심되면 `users.py passwd`로 바꿉니다.
- [ ] macOS, cloudflared(`brew upgrade cloudflared`), 이 프로젝트를 최신으로 유지합니다(아래 "업데이트").
- [ ] 정기적으로 백업합니다(아래 "백업과 복원").
- [ ] (선택) 한 겹 더 막으려면 Cloudflare Access로 이 주소 앞에 이메일 인증을 둡니다. Cloudflare One(Zero Trust) 대시보드의 Access › Applications에서 `chat.example.com`을 Self-hosted 앱으로 추가하고 친구들의 이메일만 허용하세요. 50명까지 무료입니다.

앱 자체에도 보호 장치가 있습니다. 15분 안에 같은 IP나 같은 이름으로 10번 틀리면 잠시 로그인을 막고(429), 계정 모드에서는 `/docs`와 `/openapi.json`을 숨기며, 각자 자기 대화와 파일만 볼 수 있습니다.

## 6. 백업과 복원

백업할 것은 `data/chat.db`(계정·대화·설정)와 `data/files/`(첨부 이미지)입니다. `data/logs/`는 없어도 됩니다. 이미지를 한 번도 올리지 않았다면 `data/files/`가 없을 수 있습니다.

**서버가 켜져 있을 때**: sqlite3의 `.backup` 명령은 서버가 쓰는 중에도 온전한 사본을 만듭니다. 데이터베이스를 먼저, 이미지를 나중에 복사하세요.

```sh
mkdir -p ~/pubmed-chat-backup
sqlite3 data/chat.db ".backup '$HOME/pubmed-chat-backup/chat-$(date +%Y%m%d).db'"
rsync -a data/files/ ~/pubmed-chat-backup/files/
```

서버가 켜져 있을 때 `cp data/chat.db`로 복사하지 마세요. 최근 기록이 아직 `chat.db-wal`에만 있어 빠지거나, 깨진 사본이 될 수 있습니다.

**서버를 멈추고**: 폴더째 복사하면 됩니다.

```sh
launchctl bootout gui/$(id -u)/com.pubmedchat.server     # 터미널에서 직접 실행 중이라면 Ctrl+C
cp -R data ~/pubmed-chat-backup/data-$(date +%Y%m%d)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pubmedchat.server.plist
```

Time Machine도 `data/`를 백업하지만 서버가 쓰는 도중의 사본일 수 있으니, 중요한 때는 위 방법으로 따로 남기세요.

**복원**

```sh
launchctl bootout gui/$(id -u)/com.pubmedchat.server     # 서버 멈추기
mv data data-before-restore                               # 지금 상태는 지우지 말고 옆으로 옮겨 둡니다
mkdir -m 700 data && mkdir data/logs                      # 앱이 만들 때처럼 나만 읽을 수 있는 폴더로
cp ~/pubmed-chat-backup/chat-20261009.db data/chat.db     # 복원할 백업 파일
rsync -a ~/pubmed-chat-backup/files/ data/files/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pubmedchat.server.plist
```

- 백업 파일을 기존 `data/chat.db` 위에 그냥 덮어쓰지 마세요. 옆에 남은 `chat.db-wal`·`chat.db-shm` 파일이 복원한 데이터베이스를 망가뜨릴 수 있습니다. 위처럼 `data`를 통째로 옮겨 두면 안전합니다.
- launchd는 로그 폴더를 만들어 주지 않으므로 `data/logs`도 함께 만듭니다.
- 복원하면 서버의 대화가 백업한 시점으로 돌아갑니다.

## 7. 업데이트

```sh
cd ~/workspace/any/pubmed-llm-api                                                  # 프로젝트 폴더
mkdir -p ~/pubmed-chat-backup
sqlite3 data/chat.db ".backup '$HOME/pubmed-chat-backup/chat-before-update.db'"   # 먼저 백업
git pull
uv sync --frozen
launchctl kickstart -k gui/$(id -u)/com.pubmedchat.server
curl http://127.0.0.1:8000/health
```

- `models.json`이 바뀌었다면 다시 시작하기 전에 `uv run python scripts/prepare_models.py`로 새 모델을 받습니다.
- 다시 시작하면 진행 중이던 답변은 끊깁니다. 친구들이 쓰지 않는 시간에 하세요.
- 앱은 화면 파일(`/static/`)에 `Cache-Control: no-cache`를 붙여 보내므로 Cloudflare와 브라우저가 매번 새 버전이 있는지 확인합니다. 그래도 업데이트 뒤 화면이 깨져 보이면 강력 새로고침(⌘⇧R, Ctrl+Shift+R)을 하세요. Cloudflare에 캐시 규칙(Cache Rules)을 따로 만들어 원래 서버의 캐시 설정을 무시하게 했다면, 대시보드의 Caching › Configuration에서 **Purge Everything**으로 비운 뒤 새로고침하세요.
- cloudflared는 `brew upgrade cloudflared`로 올린 뒤 다시 시작합니다. 대시보드 방식(sudo)이면 `sudo launchctl kickstart -k system/com.cloudflare.cloudflared`, 직접 관리 방식이면 `launchctl kickstart -k gui/$(id -u)/com.cloudflare.cloudflared`입니다.

## 8. 문제 해결

**Cloudflare 502 Bad gateway**: 터널은 연결됐지만 앱이 응답하지 않습니다. 앱이 꺼졌거나 아직 켜지는 중입니다. 맥에 로그인하기 전에는 앱이 꺼져 있습니다.

```sh
curl http://127.0.0.1:8000/health
launchctl print gui/$(id -u)/com.pubmedchat.server | grep -E 'state =|last exit code'
tail -n 50 data/logs/server.log
```

**Cloudflare 1033**: cloudflared가 Cloudflare에 연결되어 있지 않습니다. cloudflared가 실행 중인지, 인터넷이 되는지, 맥이 잠들지 않았는지 확인하세요. sudo로 설치한 cloudflared의 기록은 `/Library/Logs/com.cloudflare.cloudflared.err.log`에 있습니다.

```sh
sudo launchctl list | grep cloudflared     # 대시보드 방식 (sudo로 설치)
launchctl list | grep cloudflared          # 직접 관리 방식
```

**Cloudflare 524**: 앱이 125초(예전 기준 100초) 동안 아무것도 보내지 않아 Cloudflare가 연결을 끊었습니다. 스트리밍 답변은 15초마다 `: ping` 신호를 보내므로 이 오류가 나지 않습니다. API를 `"stream": false`로 불러 긴 답변을 기다릴 때 생길 수 있으니 `"stream": true`를 쓰세요.

**`address already in use` 또는 `[Errno 48]`**: 포트 8000을 다른 서버가 쓰고 있습니다. 자동 시작 서버가 켜진 상태에서 터미널로 또 실행한 경우가 많습니다.

```sh
lsof -nP -iTCP:8000 -sTCP:LISTEN     # 포트를 쓰는 프로그램과 PID
```

**서버가 켜졌다 꺼지기를 반복함**: `data/logs/server.log`의 마지막 부분에 이유가 있습니다. "로그인할 수 있는 계정이 없습니다"라면 계정을 만든 뒤 `launchctl kickstart -k gui/$(id -u)/com.pubmedchat.server`로 다시 시작하세요.

**로그인할 때 429**: 15분 안에 10번 이상 틀렸습니다. 잠시 기다리거나 `users.py passwd`로 비밀번호를 새로 정하세요.

**밤사이 접속이 안 됨**: 맥이 잠들었을 수 있습니다. 덮개와 전원을 확인하고, 잠자기 방지가 걸려 있는지 봅니다.

```sh
pmset -g assertions | grep caffeinate
```

**`Operation not permitted`**: 프로젝트가 문서·데스크톱·다운로드 폴더 안에 있다면 macOS 개인정보 보호가 자동으로 켜진 서버의 접근을 막은 것일 수 있습니다. 다른 곳으로 옮기고 설치 스크립트를 다시 실행하세요.

**로그 보기**

```sh
tail -n 100 data/logs/server.log    # 서버 시작·종료·오류
tail -f data/logs/access.log        # 접속 기록 (Ctrl+C로 그만 보기)
```
