"""Serve the app to other people: accounts required, loopback only, the Mac kept awake."""
import argparse
from contextlib import closing
import ipaddress
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn
from pydantic import ValidationError
from app.settings import Settings

DATABASE = 'chat.db'


def say(message: str) -> None:
    print(message, file=sys.stderr)


def count_accounts(database: Path) -> int:
    """Accounts that can sign in; '!' hashes, such as the implicit local user, never can."""
    if not database.is_file():
        return 0
    # Read-only, so a check never creates or changes the database a running server is using.
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'users'").fetchone() is None:
            return 0
        return db.execute("SELECT COUNT(*) FROM users WHERE substr(password_hash, 1, 1) <> '!'").fetchone()[0]


def refusal(auth_enabled: bool, accounts: int) -> str | None:
    """Why server mode must not start, or None when it may."""
    if not auth_enabled:
        return ('AUTH_ENABLED가 false로 설정되어 있어 서버 모드를 시작하지 않습니다. '
                '로그인 없이 누구나 쓸 수 있게 되기 때문입니다.\n'
                'AUTH_ENABLED 환경 변수를 지우거나 true로 바꾼 뒤 다시 실행하세요.')
    if accounts == 0:
        return ('로그인할 수 있는 계정이 없습니다. 먼저 계정을 만드세요:\n'
                '  uv run python scripts/users.py add <이름>')
    return None


def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def port_number(text: str) -> int:
    port = int(text)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('포트는 1~65535 사이여야 합니다')
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='계정 모드로 채팅 서버를 실행합니다. 자세한 설명은 docs/deploy.md에 있습니다.')
    parser.add_argument('--host', help='접속을 받을 주소 (기본 127.0.0.1: 이 PC와 이 PC의 Cloudflare Tunnel만 접속)')
    parser.add_argument('--port', type=port_number, default=8000, help='포트 (기본 8000)')
    parser.add_argument('--allow-lan', action='store_true',
                        help='0.0.0.0에 열어 같은 네트워크의 다른 기기도 직접 접속하게 합니다 (암호화되지 않음)')
    parser.add_argument('--no-caffeinate', action='store_true', help='macOS에서 잠자기를 막지 않습니다')
    parser.add_argument('--check', action='store_true', help='서버를 켜지 않고 시작할 수 있는지만 확인합니다')
    args = parser.parse_args(argv)
    if args.host is None:
        args.host = '0.0.0.0' if args.allow_lan else '127.0.0.1'
    elif not (args.allow_lan or is_loopback(args.host)):
        parser.error(f'--host {args.host}는 이 PC 밖에서도 접속할 수 있는 주소입니다. '
                     '의도한 것이라면 --allow-lan을 함께 지정하세요.')
    return args


def keep_awake(pid: int) -> None:
    """Prevent idle sleep for as long as `pid` lives; the display may still turn off."""
    if sys.platform != 'darwin':
        return
    caffeinate = shutil.which('caffeinate')
    if caffeinate is None:
        say('caffeinate를 찾지 못해 잠자기 방지 없이 실행합니다.')
        return
    # -w ties the assertion to the server process, so it ends even if the server is killed.
    subprocess.Popen([caffeinate, '-i', '-w', str(pid)], stdin=subprocess.DEVNULL)
    say('잠자기 방지(caffeinate -i)를 켰습니다. 화면은 꺼져도 되지만 노트북 덮개를 닫으면 잠듭니다.')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # app.main reads its settings when uvicorn imports it, so the environment must be final before that.
    # An environment variable also beats .env, so a local-mode .env cannot switch accounts off here.
    os.environ.setdefault('AUTH_ENABLED', 'true')
    try:
        settings = Settings()
        accounts = count_accounts(settings.data_dir / DATABASE)
    except ValidationError as exc:
        say(f'설정 값이 올바르지 않습니다:\n{exc}')
        return 2
    except sqlite3.Error as exc:
        say(f'계정 데이터베이스를 읽지 못했습니다: {exc}')
        return 1
    problem = refusal(settings.auth_enabled, accounts)
    if problem:
        say(problem)
        return 1
    if args.check:
        say(f'서버 모드로 시작할 수 있습니다 (로그인할 수 있는 계정 {accounts}개).')
        return 0
    say(f'서버 모드: 로그인할 수 있는 계정 {accounts}개, 데이터 {settings.data_dir}')
    if args.allow_lan:
        say(f'주의: {args.host}:{args.port}에 엽니다. 같은 네트워크의 다른 기기도 암호화되지 않은 http로 '
            '직접 접속할 수 있습니다. 인터넷 공개에는 이 옵션 대신 Cloudflare Tunnel을 쓰세요.')
    if not args.no_caffeinate:
        keep_awake(os.getpid())
    # One worker: each process would load its own copy of the model, and live answers are tracked in memory.
    # cloudflared connects from 127.0.0.1, so only its X-Forwarded-For/-Proto are trusted: login limits
    # see the real client address and session cookies get Secure on https.
    # An answer still streaming would otherwise hold shutdown until launchd kills the process.
    uvicorn.run('app.main:app', host=args.host, port=args.port, workers=1, proxy_headers=True,
                forwarded_allow_ips='127.0.0.1', timeout_graceful_shutdown=10)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
