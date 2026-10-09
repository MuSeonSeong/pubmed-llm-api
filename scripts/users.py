"""Manage the accounts that may sign in when the server runs with AUTH_ENABLED=true.

  uv run python scripts/users.py add <name> [--password-stdin]
  uv run python scripts/users.py passwd <name> [--password-stdin]
  uv run python scripts/users.py list
  uv run python scripts/users.py remove <name> --yes
  uv run python scripts/users.py adopt-local <name>

Changes apply to a running server at once; a new password signs the account out everywhere.
"""
import argparse
from datetime import datetime
import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.auth import check_name, check_password, hash_password, normalize_name
from app.settings import Settings
from app.storage import LOCAL_USER, Storage


def read_password(from_stdin: bool) -> str:
    if from_stdin:  # one line, for scripts: printf '%s\n' "$PASSWORD" | ... --password-stdin
        if sys.stdin.isatty():  # typed rather than piped: still never shown on screen
            return getpass.getpass('비밀번호: ')
        return sys.stdin.readline().removesuffix('\n').removesuffix('\r')
    password = getpass.getpass('비밀번호: ')
    if getpass.getpass('비밀번호 확인: ') != password:
        raise ValueError('두 비밀번호가 서로 다릅니다.')
    return password


def new_password(args) -> str:
    password = read_password(args.password_stdin)
    check_password(password)
    return password


def add(storage: Storage, args) -> int:
    name = check_name(args.name)
    if storage.find_user(name):
        raise ValueError(f"'{name}' 계정이 이미 있습니다.")
    if storage.add_user(name, hash_password(new_password(args))) is None:
        raise ValueError(f"'{name}' 계정이 이미 있습니다.")
    print(f"'{name}' 계정을 만들었습니다.")
    return 0


def passwd(storage: Storage, args) -> int:
    name = normalize_name(args.name)
    found = storage.find_user(name)
    if found is None:
        raise ValueError(f"'{name}' 계정이 없습니다.")
    if found[1].lower() == LOCAL_USER:
        raise ValueError(f"'{LOCAL_USER}'은(는) 로컬 모드용 사용자라 비밀번호를 정할 수 없습니다.")
    storage.set_password(found[1], hash_password(new_password(args)))
    print(f"'{found[1]}'의 비밀번호를 바꿨습니다. 로그인되어 있던 기기에서는 다시 로그인해야 합니다.")
    return 0


def size(count: int) -> str:
    return f'{count / 1024 / 1024:.1f} MB' if count >= 1024 * 1024 else f'{count / 1024:.0f} KB'


def list_users(storage: Storage, args) -> int:
    users = storage.users()
    print(f'데이터 위치: {storage.data_dir}')
    for user in users:
        created = datetime.fromtimestamp(user['created_at']).strftime('%Y-%m-%d %H:%M')
        note = '' if user['can_sign_in'] else ' · 로컬 모드 전용, 로그인 불가'
        print(f"- {user['name']} · 만든 날짜 {created} · 대화 {user['chats']}개 · {size(user['bytes'])}{note}")
    if not any(user['can_sign_in'] for user in users):
        print('로그인할 수 있는 계정이 없습니다. 추가: uv run python scripts/users.py add <이름>')
    return 0


def remove(storage: Storage, args) -> int:
    name = normalize_name(args.name)
    if not args.yes:
        print(f"'{name}' 계정과 그 대화·설정·이미지가 모두 지워지며 되돌릴 수 없습니다. "
              '계속하려면 --yes를 붙여 다시 실행하세요.', file=sys.stderr)
        return 1
    found = storage.find_user(name)
    if found is None or not storage.remove_user(found[1]):
        raise ValueError(f"'{name}' 계정이 없습니다.")
    print(f"'{found[1]}' 계정과 그 데이터를 지웠습니다.")
    return 0


def adopt_local(storage: Storage, args) -> int:
    name = normalize_name(args.name)
    found = storage.find_user(name)
    if found is None or found[1].lower() == LOCAL_USER:
        raise ValueError(f"'{name}' 계정이 없습니다. 먼저 add로 만들어 주세요.")
    moved, kept = storage.adopt_local(found[0])
    print(f"로컬 모드의 대화 {moved}개를 '{found[1]}' 계정으로 옮겼습니다.")
    if kept:
        print(f'같은 대화가 이미 계정에 있어 {kept}개는 계정의 것을 그대로 두었습니다.')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='서버 모드(AUTH_ENABLED=true)에서 로그인할 계정을 관리합니다.')
    commands = parser.add_subparsers(dest='command', required=True, metavar='명령')
    for command, help_text in [('add', '계정 만들기'), ('passwd', '비밀번호 바꾸기')]:
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument('name', help='계정 이름')
        sub.add_argument('--password-stdin', action='store_true', help='비밀번호를 묻지 않고 표준 입력의 첫 줄에서 읽기')
    commands.add_parser('list', help='계정 목록과 사용량 보기')
    sub = commands.add_parser('remove', help='계정과 그 데이터를 모두 지우기')
    sub.add_argument('name', help='계정 이름')
    sub.add_argument('--yes', action='store_true', help='확인 없이 지우기')
    sub = commands.add_parser('adopt-local', help='로컬 모드(로그인 없이)에서 쓰던 대화·설정·이미지를 계정으로 옮기기')
    sub.add_argument('name', help='옮겨 받을 계정 이름')
    args = parser.parse_args(argv)
    settings = Settings()
    storage = Storage(settings.data_dir, settings.user_storage_mb * 1024 * 1024)
    run = {'add': add, 'passwd': passwd, 'list': list_users, 'remove': remove, 'adopt-local': adopt_local}
    try:
        return run[args.command](storage, args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print('\n취소했습니다.', file=sys.stderr)
        return 1
    finally:
        storage.close()


if __name__ == '__main__':
    sys.exit(main())
