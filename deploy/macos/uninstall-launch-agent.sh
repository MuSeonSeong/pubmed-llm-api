#!/bin/bash
# Stop the chat server LaunchAgent and remove it; chats and accounts in data/ are left untouched.
# Usage: bash deploy/macos/uninstall-launch-agent.sh [--dry-run]
set -euo pipefail

LABEL=com.pubmedchat.server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SERVICE="gui/$(id -u)/$LABEL"
DRY_RUN=0

usage() {
  echo "사용법: bash deploy/macos/uninstall-launch-agent.sh [--dry-run]"
  echo "  --dry-run  실행할 명령만 출력하고 아무것도 바꾸지 않습니다"
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

if [ "$DRY_RUN" = 1 ]; then
  echo "launchctl bootout $SERVICE"
  echo "rm -f \"$PLIST\""
  exit 0
fi

if [ "$(uname -s)" != Darwin ]; then
  echo "이 스크립트는 macOS 전용입니다." >&2
  exit 1
fi
if [ "$(id -u)" = 0 ]; then
  echo "sudo 없이 실행하세요. 자동 시작은 로그인한 사용자에게 등록되어 있습니다." >&2
  exit 1
fi

if launchctl print "$SERVICE" >/dev/null 2>&1; then
  if launchctl bootout "$SERVICE"; then
    echo "서버를 멈췄습니다."
  else
    echo "주의: 서버를 멈추지 못했습니다. 로그아웃하거나 재시동하면 멈춥니다." >&2
  fi
fi
if [ -e "$PLIST" ]; then
  rm -f "$PLIST"
  echo "자동 시작을 해제했습니다: $PLIST"
else
  echo "등록된 자동 시작 설정이 없습니다."
fi
echo "대화와 계정(data/)은 그대로 남아 있습니다. 직접 실행하려면: uv run python scripts/serve.py"
