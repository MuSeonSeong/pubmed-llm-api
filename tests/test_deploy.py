from contextlib import closing
import os
from pathlib import Path
import plistlib
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / 'deploy' / 'macos' / 'install-launch-agent.sh'
UNINSTALL = ROOT / 'deploy' / 'macos' / 'uninstall-launch-agent.sh'
REAL = 'scrypt$16384$8$1$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaGhhc2g='
needs_bash = pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None, reason='needs a Unix shell')


def make_database(path, *hashes):
    """A chat.db with the shared users schema and one row per password hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, '
                   'password_hash TEXT NOT NULL, created_at INTEGER NOT NULL)')
        db.executemany('INSERT INTO users(name, password_hash, created_at) VALUES (?, ?, 0)',
                       [(f'user{index}', value) for index, value in enumerate(hashes)])
        db.commit()
    return path


def test_missing_database_counts_zero_and_is_not_created(tmp_path):
    from scripts.serve import count_accounts
    assert count_accounts(tmp_path / 'data' / 'chat.db') == 0
    assert not (tmp_path / 'data').exists()


def test_implicit_local_user_is_not_an_account(tmp_path):
    from scripts.serve import count_accounts
    assert count_accounts(make_database(tmp_path / 'chat.db', '!local')) == 0


def test_only_hashes_that_can_sign_in_are_counted(tmp_path):
    from scripts.serve import count_accounts
    path = make_database(tmp_path / 'odd #?% dir' / 'chat.db', '!local', REAL, '!disabled', REAL)
    assert count_accounts(path) == 2


def test_database_without_users_table_counts_zero(tmp_path):
    from scripts.serve import count_accounts
    path = tmp_path / 'chat.db'
    path.write_bytes(b'')
    assert count_accounts(path) == 0


def test_counting_does_not_wait_for_a_running_server(tmp_path):
    from scripts.serve import count_accounts
    path = make_database(tmp_path / 'chat.db', REAL)
    with closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as server:
        server.execute('BEGIN IMMEDIATE')
        server.execute("INSERT INTO users(name, password_hash, created_at) VALUES ('later', ?, 0)", (REAL,))
        assert count_accounts(path) == 1
        server.execute('ROLLBACK')


def test_accounts_are_counted_in_the_database_the_app_writes(tmp_path):
    """The launcher reads what app.storage and scripts/users.py write, with the app holding it open or not."""
    from app.auth import hash_password
    from app.storage import Storage
    from scripts.serve import count_accounts
    storage = Storage(tmp_path, 1024 * 1024)
    storage.local_user()
    assert count_accounts(tmp_path / 'chat.db') == 0
    storage.add_user('민수', hash_password('password123'))
    assert count_accounts(tmp_path / 'chat.db') == 1
    storage.close()  # the last connection removes the WAL files; a read-only check must still work
    assert count_accounts(tmp_path / 'chat.db') == 1


def test_refusal_reasons():
    from scripts.serve import refusal
    assert 'AUTH_ENABLED' in refusal(False, 3)
    assert 'scripts/users.py add' in refusal(True, 0)
    assert refusal(True, 1) is None


def test_arguments_default_to_loopback():
    from scripts.serve import parse_args
    args = parse_args([])
    assert (args.host, args.port, args.allow_lan, args.no_caffeinate, args.check) == \
        ('127.0.0.1', 8000, False, False, False)
    assert parse_args(['--allow-lan']).host == '0.0.0.0'
    assert parse_args(['--host', '192.168.0.5', '--allow-lan']).host == '192.168.0.5'
    assert parse_args(['--host', 'localhost', '--port', '8771']).port == 8771
    assert parse_args(['--host', '::1']).host == '::1'


@pytest.mark.parametrize('argv', [['--host', '0.0.0.0'], ['--host', '192.168.0.5'], ['--host', 'example.com'],
                                  ['--port', '0'], ['--port', '65536'], ['--port', 'http']])
def test_arguments_that_expose_the_server_or_are_invalid_exit(argv):
    from scripts.serve import parse_args
    with pytest.raises(SystemExit) as exit_info:
        parse_args(argv)
    assert exit_info.value.code == 2


@pytest.fixture
def server(tmp_path, monkeypatch):
    """serve.main with settings, uvicorn and caffeinate replaced by recorders; AUTH_ENABLED starts unset."""
    from scripts import serve
    import uvicorn
    # setenv first so that monkeypatch restores the variable that main() sets.
    monkeypatch.setenv('AUTH_ENABLED', '')
    monkeypatch.delenv('AUTH_ENABLED')
    record = SimpleNamespace(env_at_settings=None, run=None, awake=None, data=tmp_path / 'data')

    def settings():
        record.env_at_settings = os.environ.get('AUTH_ENABLED')
        return SimpleNamespace(auth_enabled=os.environ['AUTH_ENABLED'].lower() in ('1', 'true', 'yes', 'on'),
                               data_dir=record.data)

    def run(app, **options):
        record.run = (app, options)

    def awake(pid):
        record.awake = pid
    monkeypatch.setattr(serve, 'Settings', settings)
    monkeypatch.setattr(uvicorn, 'run', run)
    monkeypatch.setattr(serve, 'keep_awake', awake)
    return serve, record


def test_server_mode_turns_accounts_on_before_reading_settings(server):
    serve, record = server
    make_database(record.data / 'chat.db', '!local', REAL)
    assert serve.main([]) == 0
    assert record.env_at_settings == 'true' and os.environ['AUTH_ENABLED'] == 'true'
    app, options = record.run
    # An import string, so app.main reads its settings only after the environment above is set.
    assert app == 'app.main:app'
    assert options == {'host': '127.0.0.1', 'port': 8000, 'workers': 1, 'proxy_headers': True,
                       'forwarded_allow_ips': '127.0.0.1', 'timeout_graceful_shutdown': 10}
    assert record.awake == os.getpid()


def test_no_account_refuses_and_explains_how_to_add_one(server, capsys):
    serve, record = server
    make_database(record.data / 'chat.db', '!local')
    assert serve.main([]) == 1
    assert 'uv run python scripts/users.py add' in capsys.readouterr().err
    assert record.run is None and record.awake is None


def test_missing_database_refuses_without_creating_it(server, capsys):
    serve, record = server
    assert serve.main([]) == 1
    assert 'scripts/users.py add' in capsys.readouterr().err
    assert not record.data.exists() and record.run is None


def test_explicit_auth_disabled_is_refused(server, monkeypatch, capsys):
    serve, record = server
    make_database(record.data / 'chat.db', REAL)
    monkeypatch.setenv('AUTH_ENABLED', 'false')
    assert serve.main([]) == 1
    assert os.environ['AUTH_ENABLED'] == 'false'
    assert 'AUTH_ENABLED' in capsys.readouterr().err
    assert record.run is None and record.awake is None


def test_check_reports_without_starting(server, capsys):
    serve, record = server
    make_database(record.data / 'chat.db', REAL, REAL)
    assert serve.main(['--check']) == 0
    assert '2개' in capsys.readouterr().err
    assert record.run is None and record.awake is None


def test_lan_binding_warns_and_caffeinate_can_be_skipped(server, capsys):
    serve, record = server
    make_database(record.data / 'chat.db', REAL)
    assert serve.main(['--allow-lan', '--no-caffeinate', '--port', '8771']) == 0
    assert record.run[1]['host'] == '0.0.0.0' and record.run[1]['port'] == 8771
    assert '주의' in capsys.readouterr().err
    assert record.awake is None


def test_invalid_settings_exit_without_traceback(server, monkeypatch, capsys):
    from pydantic import BaseModel
    serve, record = server

    class Strict(BaseModel):
        session_days: int
    monkeypatch.setattr(serve, 'Settings', lambda: Strict(session_days='often'))
    assert serve.main([]) == 2
    assert '설정' in capsys.readouterr().err
    assert record.run is None


def test_unreadable_database_exits_without_traceback(server, monkeypatch, capsys):
    serve, record = server
    record.data.mkdir()
    (record.data / 'chat.db').write_bytes(b'not a database ' * 64)
    assert serve.main([]) == 1
    assert '데이터베이스' in capsys.readouterr().err
    assert record.run is None


def test_caffeinate_follows_the_server_process_on_macos_only(monkeypatch):
    from scripts import serve
    started = []
    monkeypatch.setattr(serve.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(serve.subprocess, 'Popen', lambda command, **options: started.append(command))
    monkeypatch.setattr(serve.sys, 'platform', 'darwin')
    serve.keep_awake(4321)
    assert started == [['/usr/bin/caffeinate', '-i', '-w', '4321']]
    monkeypatch.setattr(serve.sys, 'platform', 'linux')
    serve.keep_awake(4321)
    assert len(started) == 1


def test_importing_the_launcher_does_not_import_the_app():
    # app.main builds the app, reading settings, on import; that must wait until main() set AUTH_ENABLED.
    code = 'import sys, scripts.serve; print("app.main" in sys.modules)'
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'False'


def fake_tools(tmp_path):
    """A PATH whose uv and launchctl fail loudly, and an empty HOME, so nothing real is touched."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('uv', 'launchctl'):
        tool = bin_dir / name
        tool.write_text('#!/bin/sh\necho "unexpected call: $0 $*" >&2\nexit 99\n')
        tool.chmod(0o755)
    home = tmp_path / 'home'
    home.mkdir()
    return {'PATH': f'{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(home)}, bin_dir, home


@needs_bash
def test_dry_run_prints_a_valid_plist_and_changes_nothing(tmp_path):
    project = tmp_path / 'my & <chat> project'
    script = project / 'deploy' / 'macos' / INSTALL.name
    script.parent.mkdir(parents=True)
    shutil.copy(INSTALL, script)
    env, bin_dir, home = fake_tools(tmp_path)
    result = subprocess.run(['bash', str(script), '--dry-run'], env=env, capture_output=True, check=True)
    plist = plistlib.loads(result.stdout)
    root = str(project.resolve())
    assert plist['Label'] == 'com.pubmedchat.server'
    assert plist['ProgramArguments'] == [str(bin_dir / 'uv'), 'run', 'python', 'scripts/serve.py']
    assert plist['WorkingDirectory'] == root
    assert plist['EnvironmentVariables']['PATH'].startswith(f'{bin_dir}:')
    assert (plist['RunAtLoad'], plist['KeepAlive'], plist['ThrottleInterval']) == (True, True, 30)
    assert plist['StandardOutPath'] == root + '/data/logs/access.log'
    assert plist['StandardErrorPath'] == root + '/data/logs/server.log'
    assert b'unexpected call' not in result.stderr
    assert not (project / 'data').exists()
    assert list(home.iterdir()) == []


@needs_bash
@pytest.mark.parametrize('script', [INSTALL, UNINSTALL])
def test_scripts_reject_unknown_options_before_doing_anything(tmp_path, script):
    env, _, home = fake_tools(tmp_path)
    result = subprocess.run(['bash', str(script), '--force'], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'unexpected call' not in result.stderr
    assert list(home.iterdir()) == []


@needs_bash
def test_uninstall_dry_run_only_prints_commands(tmp_path):
    env, _, home = fake_tools(tmp_path)
    result = subprocess.run(['bash', str(UNINSTALL), '--dry-run'], env=env, capture_output=True, text=True,
                            check=True)
    assert f'launchctl bootout gui/{os.getuid()}/com.pubmedchat.server' in result.stdout
    assert 'unexpected call' not in result.stderr
    assert list(home.iterdir()) == []
