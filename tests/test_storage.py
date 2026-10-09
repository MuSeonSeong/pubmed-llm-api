import json
import sqlite3
import threading
import pytest
from app.settings import ServiceError


class Clock:
    def __init__(self, now=1_760_000_000):
        self.now = now
    def __call__(self):
        return self.now


def storage(data_dir, quota=10_000_000, clock=None):
    from app.storage import Storage
    return Storage(data_dir, quota, clock or Clock())


def chat(chat_id, rev, text='hello'):
    return json.dumps({'id': chat_id, 'rev': rev, 'title': '새 대화', 'messages': [{'role': 'user', 'content': text}]},
                      ensure_ascii=False)


def test_database_is_created_on_first_use_with_wal_and_schema(data_dir):
    s = storage(data_dir)
    assert not data_dir.exists()
    s.local_user()
    db = sqlite3.connect(data_dir / 'chat.db')
    assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert db.execute('PRAGMA user_version').fetchone()[0] == 1
    assert {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")} == {
        'users', 'sessions', 'chats', 'settings', 'files', 'meta'}
    assert db.execute("SELECT name, password_hash FROM users").fetchall() == [('local', '!local')]
    assert s._db.execute('PRAGMA foreign_keys').fetchone()[0] == 1
    assert s.local_user() == s.local_user()  # created once


def test_revisions_and_sequences_report_exactly_what_changed(data_dir):
    s = storage(data_dir)
    user = s.local_user()
    rev, first = s.put_chat(user, 'a', 1, chat('a', 1))
    assert rev == 1
    _, second = s.put_chat(user, 'b', 5, chat('b', 5, '두 번째'))
    assert second > first
    seq, chats, deleted = s.changes(user, 0)
    assert seq == second and [json.loads(c)['id'] for c in chats] == ['a', 'b'] and deleted == []
    assert s.changes(user, first)[1] == [chat('b', 5, '두 번째')]  # stored verbatim
    assert s.put_chat(user, 'a', 1, chat('a', 1, 'same rev')) == (1, first)  # an equal revision changes nothing
    assert json.loads(s.changes(user, 0)[1][0])['messages'][0]['content'] == 'hello'
    _, third = s.put_chat(user, 'a', 2, chat('a', 2, 'edited'))
    seq, chats, _ = s.changes(user, second)
    assert seq == third and [json.loads(c)['messages'][0]['content'] for c in chats] == ['edited']
    assert s.changes(user, third) == (third, [], [])


def test_an_older_revision_is_refused_with_the_stored_chat(data_dir):
    from app.storage import StaleChat
    s = storage(data_dir)
    user = s.local_user()
    s.put_chat(user, 'a', 5, chat('a', 5, 'newer'))
    with pytest.raises(StaleChat) as e:
        s.put_chat(user, 'a', 4, chat('a', 4, 'older'))
    assert json.loads(e.value.data)['messages'][0]['content'] == 'newer'


def test_deletion_wins_over_any_later_revision(data_dir):
    s = storage(data_dir)
    user = s.local_user()
    _, before = s.put_chat(user, 'a', 1, chat('a', 1))
    s.delete_chat(user, 'a')
    seq, chats, deleted = s.changes(user, before)
    assert chats == [] and deleted == ['a'] and seq > before
    with pytest.raises(ServiceError) as e:
        s.put_chat(user, 'a', 10 ** 12, chat('a', 10 ** 12))
    assert (e.value.code, e.value.status_code) == ('chat_deleted', 410)
    s.delete_chat(user, 'a')  # deleting again is not a new change
    assert s.changes(user, seq) == (seq, [], [])
    s.delete_chat(user, 'never-saved')  # a tombstone even for a chat the server never had
    assert s.changes(user, seq)[2] == ['never-saved']
    assert s.changes(user, 0)[1:] == ([], ['a', 'never-saved'])


def test_each_user_sees_only_their_own_chats(data_dir):
    s = storage(data_dir)
    alice, bob = s.add_user('alice', 'scrypt$x'), s.add_user('bob', 'scrypt$y')
    s.put_chat(alice, 'same-id', 1, chat('same-id', 1, 'alice'))
    s.put_chat(bob, 'same-id', 1, chat('same-id', 1, 'bob'))
    s.delete_chat(bob, 'other')
    assert [json.loads(c)['messages'][0]['content'] for c in s.changes(alice, 0)[1]] == ['alice']
    assert s.changes(alice, 0)[2] == []


def test_quota_counts_chat_bytes_and_files_but_lets_a_chat_shrink(data_dir):
    s = storage(data_dir, quota=1000)
    user = s.local_user()
    s.put_chat(user, 'a', 1, chat('a', 1, '가' * 150))  # Hangul is 3 bytes in UTF-8
    assert s.usage(user) == len(chat('a', 1, '가' * 150).encode())
    with pytest.raises(ServiceError) as e:
        s.put_chat(user, 'b', 1, chat('b', 1, 'x' * 600))
    assert (e.value.code, e.value.status_code) == ('storage_quota', 413)
    with pytest.raises(ServiceError):
        s.add_file(user, 'image/png', b'x' * 600)
    assert not [path for path in (data_dir / 'files').rglob('*') if path.is_file()]  # nothing left behind
    s.put_chat(user, 'a', 2, chat('a', 2, 'short'))
    file_id = s.add_file(user, 'image/png', b'x' * 600)
    assert s.usage(user) == len(chat('a', 2, 'short').encode()) + 600
    assert s.read_file(user, file_id) == ('image/png', b'x' * 600)


def test_full_storage_collects_old_uploads_no_chat_uses(data_dir):
    from app.storage import FILE_GRACE_SECONDS
    clock = Clock()
    s = storage(data_dir, quota=2000, clock=clock)
    user = s.local_user()
    orphan, used, young = (s.add_file(user, 'image/png', b'o' * 500), s.add_file(user, 'image/png', b'u' * 500), None)
    s.put_chat(user, 'a', 1, json.dumps({'id': 'a', 'rev': 1, 'messages': [{'attachments': [
        {'name': 'x.png', 'kind': 'image', 'fileId': used, 'url': '/api/files/' + used}]}]}))
    clock.now += FILE_GRACE_SECONDS + 1
    young = s.add_file(user, 'image/png', b'y' * 500)  # unreferenced, but just uploaded
    s.add_file(user, 'image/png', b'n' * 500)  # does not fit until the old orphan goes
    assert s.find_file(user, orphan) is None and not s.file_path(orphan).exists()
    assert s.find_file(user, used) and s.find_file(user, young)
    with pytest.raises(ServiceError):
        s.add_file(user, 'image/png', b'z' * 500)  # nothing more may be collected


def test_a_collection_that_found_nothing_waits_for_a_chat_change_or_time(data_dir):
    from app.storage import FILE_GRACE_SECONDS, RECOLLECT_SECONDS
    clock = Clock()
    s = storage(data_dir, quota=1000, clock=clock)
    user = s.local_user()
    s.add_file(user, 'image/png', b'a' * 450)  # never referenced by any chat
    clock.now += FILE_GRACE_SECONDS - 1
    s.add_file(user, 'image/png', b'b' * 450)
    with pytest.raises(ServiceError):
        s.add_file(user, 'image/png', b'c' * 450)  # 'a' is a second too young to collect
    clock.now += 2
    with pytest.raises(ServiceError):
        s.add_file(user, 'image/png', b'c' * 450)  # 'a' is garbage now, but the last look found nothing
    s.put_chat(user, 'x', 1, chat('x', 1))  # a chat change may free images, so collection runs again
    s.add_file(user, 'image/png', b'c' * 400)
    with pytest.raises(ServiceError):
        s.add_file(user, 'image/png', b'd' * 400)
    clock.now += FILE_GRACE_SECONDS + RECOLLECT_SECONDS
    s.add_file(user, 'image/png', b'd' * 400)  # time passed: 'b' and 'c' became collectable


def test_files_belong_to_their_uploader_and_ids_are_never_paths(data_dir):
    s = storage(data_dir)
    alice, bob = s.add_user('alice', 'scrypt$x'), s.add_user('bob', 'scrypt$y')
    file_id = s.add_file(alice, 'image/gif', b'GIF89a')
    path = s.file_path(file_id)
    assert path == data_dir / 'files' / file_id[:2] / file_id and path.read_bytes() == b'GIF89a'
    assert len(file_id) == 24 and s.find_file(alice, file_id) == ('image/gif', path)
    assert s.find_file(bob, file_id) is None and s.read_file(bob, file_id) is None
    for bad in ['../chat.db', '..', '', 'a/b', 'x' * 65]:
        assert s.find_file(alice, bad) is None
    path.unlink()
    assert s.find_file(alice, file_id) is None  # a missing file is not found rather than an error


def test_sessions_expire_and_are_cleaned_up(data_dir):
    clock = Clock()
    s = storage(data_dir, clock=clock)
    user = s.add_user('alice', 'scrypt$x')
    s.add_session(user, 'digest', clock.now + 60)
    assert s.session_user('digest') == (user, 'alice')
    assert s.session_user('other') is None
    clock.now += 60
    assert s.session_user('digest') is None
    assert s._db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] == 0
    s.add_session(user, 'old', clock.now + 1)
    clock.now += 5
    s.add_session(user, 'new', clock.now + 60)  # signing in sweeps expired sessions
    assert [row[0] for row in s._db.execute('SELECT token_hash FROM sessions')] == ['new']


def test_users_without_a_password_hash_cannot_hold_sessions(data_dir):
    s = storage(data_dir)
    local = s.local_user()
    s.add_session(local, 'digest', s.now() + 60)
    assert s.session_user('digest') is None
    assert s.account_count() == 0
    s.add_user('alice', 'scrypt$x')
    assert s.account_count() == 1


def test_names_are_unique_without_regard_to_ascii_case(data_dir):
    s = storage(data_dir)
    assert s.add_user('Alice', 'scrypt$x')
    assert s.add_user('alice', 'scrypt$y') is None
    assert s.find_user('ALICE')[1] == 'Alice'
    assert s.add_user('홍길동', 'scrypt$z')


def test_new_password_and_removal_end_everything_the_user_had(data_dir):
    s = storage(data_dir)
    user = s.add_user('alice', 'scrypt$old')
    s.add_session(user, 'digest', s.now() + 60)
    assert s.set_password('ALICE', 'scrypt$new') and s.session_user('digest') is None
    assert s.find_user('alice')[2] == 'scrypt$new'
    assert not s.set_password('nobody', 'scrypt$x')
    s.add_session(user, 'again', s.now() + 60)
    s.put_chat(user, 'a', 1, chat('a', 1))
    s.put_settings(user, '{"theme":"dark"}')
    path = s.file_path(s.add_file(user, 'image/png', b'png'))
    assert s.remove_user('alice') and not s.remove_user('alice')
    assert not path.exists()
    for table in ['users', 'sessions', 'chats', 'settings', 'files']:
        assert s._db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_settings_are_last_write_wins(data_dir):
    s = storage(data_dir)
    user = s.local_user()
    assert s.settings(user) is None
    s.put_settings(user, '{"a":1}')
    s.put_settings(user, '{"a":2}')
    assert s.settings(user) == '{"a":2}'


def test_concurrent_writes_get_distinct_increasing_sequences(data_dir):
    s = storage(data_dir)
    user = s.local_user()
    seqs, errors = [], []
    def write(worker):
        try:
            for rev in range(1, 21):
                seqs.append(s.put_chat(user, f'chat{worker}', rev, chat(f'chat{worker}', rev))[1])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(set(seqs)) == 120
    seq, chats, _ = s.changes(user, 0)
    assert seq == max(seqs) and sorted(json.loads(c)['rev'] for c in chats) == [20] * 6


def test_a_second_process_sees_changes_at_once(data_dir):
    first, second = storage(data_dir), storage(data_dir)
    user = first.add_user('alice', 'scrypt$x')
    first.add_session(user, 'digest', first.now() + 60)
    assert second.session_user('digest') == (user, 'alice')
    second.set_password('alice', 'scrypt$new')  # e.g. scripts/users.py while the server runs
    assert first.session_user('digest') is None


def test_adopting_local_data_moves_chats_and_leaves_tombstones(data_dir):
    s = storage(data_dir)
    local, alice = s.local_user(), s.add_user('alice', 'scrypt$x')
    s.put_chat(local, 'mine', 3, chat('mine', 3, 'from local mode'))
    s.put_chat(local, 'both', 1, chat('both', 1, 'local copy'))
    s.put_chat(alice, 'both', 1, chat('both', 1, 'account copy'))
    s.put_settings(local, '{"theme":"dark"}')
    file_id = s.add_file(local, 'image/png', b'png')
    _, before = s.put_chat(alice, 'x', 1, chat('x', 1))
    assert s.adopt_local(alice) == (1, 1)
    _, chats, _ = s.changes(alice, before)
    assert [json.loads(c)['messages'][0]['content'] for c in chats] == ['from local mode']
    assert s.find_file(alice, file_id) and s.find_file(local, file_id) is None
    assert s.settings(alice) == '{"theme":"dark"}'
    _, chats, deleted = s.changes(local, 0)
    assert deleted == ['mine'] and [json.loads(c)['messages'][0]['content'] for c in chats] == ['local copy']
    assert s.adopt_local(alice) == (0, 1)
