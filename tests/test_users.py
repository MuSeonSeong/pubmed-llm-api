import io
import json
import os
import subprocess
import sys
import pytest
from app.catalog import ROOT

PASSWORD = 'a long secret'


def run(monkeypatch, *argv, stdin=None, prompts=None):
    from scripts import users
    if stdin is not None:
        monkeypatch.setattr(sys, 'stdin', io.StringIO(stdin))
    if prompts is not None:
        answers = iter(prompts)
        monkeypatch.setattr('getpass.getpass', lambda prompt='': next(answers))
    return users.main(list(argv))


def stored(data_dir):
    """The same database the script uses, opened the way the server opens it."""
    from app.storage import Storage
    return Storage(data_dir, 10_000_000)


def test_add_reads_the_password_from_stdin_and_never_prints_it(monkeypatch, capsys, data_dir):
    from app.auth import verify_password
    assert run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD + '\n') == 0
    out = capsys.readouterr()
    assert 'alice' in out.out and PASSWORD not in out.out + out.err
    user = stored(data_dir).find_user('alice')
    assert user[2].startswith('scrypt$') and verify_password(PASSWORD, user[2])
    assert (data_dir / 'chat.db').is_file()  # Settings().data_dir, i.e. DATA_DIR


def test_add_asks_twice_and_refuses_a_mismatch(monkeypatch, capsys, data_dir):
    assert run(monkeypatch, 'add', '홍길동', prompts=[PASSWORD, PASSWORD + 'x']) == 1
    assert '다릅니다' in capsys.readouterr().err
    assert stored(data_dir).find_user('홍길동') is None
    assert run(monkeypatch, 'add', '홍길동', prompts=[PASSWORD, PASSWORD]) == 0


@pytest.mark.parametrize('name,password', [('local', PASSWORD), ('a b', PASSWORD), ('x' * 33, PASSWORD),
                                           ('alice', 'short')])
def test_add_refuses_bad_names_and_short_passwords(monkeypatch, capsys, data_dir, name, password):
    assert run(monkeypatch, 'add', name, '--password-stdin', stdin=password + '\n') == 1
    assert capsys.readouterr().err.strip()
    assert stored(data_dir).account_count() == 0


def test_add_refuses_a_taken_name_in_any_case(monkeypatch, capsys):
    assert run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD) == 0
    assert run(monkeypatch, 'add', 'Alice', '--password-stdin', stdin=PASSWORD) == 1
    assert '이미 있습니다' in capsys.readouterr().err


def test_passwd_replaces_the_hash_and_signs_out_everywhere(monkeypatch, data_dir):
    from app.auth import verify_password
    run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD)
    s = stored(data_dir)
    user_id = s.find_user('alice')[0]
    s.add_session(user_id, 'digest', s.now() + 60)
    assert run(monkeypatch, 'passwd', 'ALICE', '--password-stdin', stdin='a new secret\r\n') == 0
    assert verify_password('a new secret', s.find_user('alice')[2])
    assert s.session_user('digest') is None
    assert run(monkeypatch, 'passwd', 'nobody', '--password-stdin', stdin=PASSWORD) == 1
    s.local_user()
    assert run(monkeypatch, 'passwd', 'local', '--password-stdin', stdin=PASSWORD) == 1


def test_list_shows_accounts_and_usage(monkeypatch, capsys, data_dir):
    assert run(monkeypatch, 'list') == 0
    assert '계정이 없습니다' in capsys.readouterr().out
    run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD)
    s = stored(data_dir)
    s.put_chat(s.local_user(), 'c1', 1, json.dumps({'id': 'c1', 'rev': 1}))
    capsys.readouterr()
    assert run(monkeypatch, 'list') == 0
    out = capsys.readouterr().out
    assert '- alice' in out and '- local' in out and '로그인 불가' in out and str(data_dir) in out
    assert '계정이 없습니다' not in out


def test_remove_needs_yes_and_deletes_everything(monkeypatch, capsys, data_dir):
    run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD)
    s = stored(data_dir)
    path = s.file_path(s.add_file(s.find_user('alice')[0], 'image/png', b'png'))
    assert run(monkeypatch, 'remove', 'alice') == 1
    assert '--yes' in capsys.readouterr().err and s.find_user('alice')
    assert run(monkeypatch, 'remove', 'alice', '--yes') == 0
    assert s.find_user('alice') is None and not path.exists()
    assert run(monkeypatch, 'remove', 'alice', '--yes') == 1


def test_adopt_local_hands_local_mode_chats_to_an_account(monkeypatch, capsys, data_dir):
    s = stored(data_dir)
    s.put_chat(s.local_user(), 'c1', 1, json.dumps({'id': 'c1', 'rev': 1}))
    assert run(monkeypatch, 'adopt-local', 'alice') == 1  # the account must exist first
    run(monkeypatch, 'add', 'alice', '--password-stdin', stdin=PASSWORD)
    capsys.readouterr()
    assert run(monkeypatch, 'adopt-local', 'alice') == 0
    assert '1개' in capsys.readouterr().out
    assert [json.loads(chat)['id'] for chat in s.changes(s.find_user('alice')[0], 0)[1]] == ['c1']


def test_script_runs_from_the_project_root(data_dir):
    env = {**os.environ, 'DATA_DIR': str(data_dir)}
    added = subprocess.run([sys.executable, 'scripts/users.py', 'add', 'alice', '--password-stdin'], cwd=ROOT, env=env,
                           input=PASSWORD + '\n', capture_output=True, text=True, timeout=120)
    assert added.returncode == 0, added.stderr
    listed = subprocess.run([sys.executable, 'scripts/users.py', 'list'], cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=120)
    assert listed.returncode == 0 and '- alice' in listed.stdout
    usage = subprocess.run([sys.executable, 'scripts/users.py'], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=120)
    assert usage.returncode == 2
