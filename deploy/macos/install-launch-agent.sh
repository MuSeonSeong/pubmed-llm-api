#!/bin/bash
# Start the chat server whenever this user logs in, and restart it if it stops (a macOS LaunchAgent).
# Usage: bash deploy/macos/install-launch-agent.sh [--dry-run]
set -euo pipefail

LABEL=com.pubmedchat.server
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$ROOT/data/logs"
DRY_RUN=0

usage() {
  echo "사용법: bash deploy/macos/install-launch-agent.sh [--dry-run]"
  echo "  --dry-run  만들 설정 파일(plist)만 출력하고 아무것도 바꾸지 않습니다"
}

if [ $# -gt 1 ]; then
  usage >&2
  exit 2
fi
case "${1:-}" in
  '') ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "알 수 없는 옵션: $1" >&2; usage >&2; exit 2 ;;
esac

# launchd starts jobs with a minimal PATH, so the plist names uv by its absolute path.
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  echo "uv를 찾을 수 없습니다. https://docs.astral.sh/uv/ 에서 설치한 뒤 다시 실행하세요." >&2
  exit 1
fi
case "$UV" in
  /*) ;;
  *) UV="$(cd "$(dirname "$UV")" && pwd -P)/$(basename "$UV")" ;;
esac

xml() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

# ProcessType Interactive: without it launchd throttles the job's CPU and I/O like a background task,
# which slows model loading and answers.
plist() {
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$(xml "$UV")</string>
		<string>run</string>
		<string>python</string>
		<string>scripts/serve.py</string>
	</array>
	<key>WorkingDirectory</key>
	<string>$(xml "$ROOT")</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key>
		<string>$(xml "$(dirname "$UV"):/usr/bin:/bin:/usr/sbin:/sbin")</string>
	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>ThrottleInterval</key>
	<integer>30</integer>
	<key>ProcessType</key>
	<string>Interactive</string>
	<key>StandardOutPath</key>
	<string>$(xml "$LOGS/access.log")</string>
	<key>StandardErrorPath</key>
	<string>$(xml "$LOGS/server.log")</string>
</dict>
</plist>
EOF
}

if [ "$DRY_RUN" = 1 ]; then
  plist
  echo "(--dry-run: 아무것도 바꾸지 않았습니다. 실제 설치 위치는 $PLIST 입니다.)" >&2
  exit 0
fi

if [ "$(uname -s)" != Darwin ]; then
  echo "이 스크립트는 macOS 전용입니다." >&2
  exit 1
fi
if [ "$(id -u)" = 0 ]; then
  echo "sudo 없이 실행하세요. 서버는 로그인한 사용자 권한으로 실행됩니다." >&2
  exit 1
fi
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"

# Refuse before touching launchd: without an account the agent would only restart in a loop.
if ! (cd "$ROOT" && "$UV" run python scripts/serve.py --check); then
  echo "자동 시작을 등록하지 않았습니다. 위 안내대로 준비한 뒤 다시 실행하세요." >&2
  exit 1
fi

port_busy=0
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "이미 등록된 자동 시작을 새 설정으로 바꿉니다."
  launchctl bootout "$SERVICE" 2>/dev/null || true
  # Bootstrapping fails while launchd still holds the old job, so wait until it is gone.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    launchctl print "$SERVICE" >/dev/null 2>&1 || break
    sleep 1
  done
elif pids="$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null | xargs)" && [ -n "$pids" ]; then
  port_busy=1
  echo "주의: 포트 8000을 이미 다른 프로그램(PID $pids)이 쓰고 있습니다." \
    "그 프로그램을 끄면 30초 안에 서버가 자동으로 켜집니다." >&2
fi

# launchd does not create missing directories for the log files.
mkdir -p "$LOGS" "$(dirname "$PLIST")"
tmp="$(mktemp "$PLIST.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
plist > "$tmp"
plutil -lint -s "$tmp"
chmod 644 "$tmp"
mv "$tmp" "$PLIST"
trap - EXIT

if ! launchctl bootstrap "$DOMAIN" "$PLIST"; then
  echo "launchd에 등록하지 못했습니다. 이 맥에 로그인한 화면의 터미널에서 sudo 없이 실행했는지 확인하세요." >&2
  exit 1
fi
launchctl kickstart -k "$SERVICE"
echo "자동 시작을 등록했습니다. 로그인하면 서버가 켜지고, 서버가 꺼지면 30초 안에 다시 켜집니다."

if [ "$port_busy" = 0 ]; then
  echo "서버가 켜지기를 기다립니다(최대 1분)..."
  ready=0
  for _ in $(seq 1 60); do
    if curl -fsS -o /dev/null --max-time 2 http://127.0.0.1:8000/health 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" = 1 ]; then
    echo "서버가 http://127.0.0.1:8000 에서 응답합니다."
  else
    echo "아직 응답이 없습니다. 로그를 확인하세요: tail -n 50 \"$LOGS/server.log\"" >&2
  fi
fi
cat <<EOF
  상태: launchctl print $SERVICE | grep -E 'state =|pid =|last exit code'
  로그: tail -f "$LOGS/server.log"
  해제: bash deploy/macos/uninstall-launch-agent.sh
EOF
