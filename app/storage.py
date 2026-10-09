"""Accounts, sessions and synced chats in DATA_DIR/chat.db, with uploaded images beside it.

Every change to a chat takes the next value of one global sequence, so a client that remembers the
last sequence it saw can ask for exactly what changed since. Nothing touches the disk until first use.
"""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from app.settings import ServiceError

SCHEMA_VERSION = 1
SCHEMA = '''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
CREATE TABLE IF NOT EXISTS chats (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL,
    rev INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    data TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, chat_id)
);
CREATE INDEX IF NOT EXISTS chats_seq ON chats(user_id, seq);
CREATE TABLE IF NOT EXISTS settings (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_user ON files(user_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('seq', 0);
'''
LOCAL_USER = 'local'
LOCAL_HASH = '!local'  # a hash starting with '!' never matches any password
FILE_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
# How chats point at uploads: attachments carry both the URL and the id.
FILE_REFERENCE = re.compile(r'(?:/api/files/|"fileId"\s*:\s*")([A-Za-z0-9_-]{1,64})')
# The client uploads an image as soon as it is attached and saves the chat that uses it later,
# so only uploads older than this can be garbage.
FILE_GRACE_SECONDS = 24 * 3600
# A collection reads every chat the user has, so one that found nothing is not repeated for a while
# unless the user's chats change; a client retrying a write that does not fit cannot stall the server.
RECOLLECT_SECONDS = 300


def stored_bytes(column: str) -> str:
    """SQL for the UTF-8 size of a text column; octet_length reads only the record header."""
    if sqlite3.sqlite_version_info >= (3, 43):
        return f'octet_length({column})'
    return f'length(CAST({column} AS BLOB))'


def quota_exceeded() -> ServiceError:
    return ServiceError('storage_quota', 'Your storage is full; delete chats with images to free space', 413)


class StaleChat(Exception):
    """The server holds a newer revision of the chat; `data` is its stored JSON."""
    def __init__(self, data: str):
        super().__init__('A newer revision of this chat is stored')
        self.data = data


class _OverQuota(Exception):
    pass


class Storage:
    """One connection guarded by a lock, so it may be called from any worker thread."""

    def __init__(self, data_dir: Path, quota_bytes: int, clock: Callable[[], float] = time.time):
        self.data_dir, self.quota_bytes, self.clock = Path(data_dir), quota_bytes, clock
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._fruitless: dict[int, float] = {}  # user id -> when a collection last freed nothing

    def now(self) -> int:
        return int(self.clock())

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Autocommit mode: every transaction below is explicit.
            db = sqlite3.connect(self.data_dir / 'chat.db', timeout=10, isolation_level=None,
                                 check_same_thread=False)
            try:
                db.execute('PRAGMA journal_mode = WAL')
                # With WAL this never corrupts the file; a power cut may only lose the last few saves,
                # which the browsers still hold and send again.
                db.execute('PRAGMA synchronous = NORMAL')
                db.execute('PRAGMA foreign_keys = ON')
                version = db.execute('PRAGMA user_version').fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise RuntimeError(f'{self.data_dir / "chat.db"} was written by a newer version of this app')
                if version < SCHEMA_VERSION:
                    db.executescript(f'BEGIN IMMEDIATE;{SCHEMA}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;')
            except BaseException:
                db.close()
                raise
            self._db = db
        return self._db

    @contextmanager
    def _transaction(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        # IMMEDIATE takes the write lock up front, so sequence numbers commit in the order they were
        # assigned even when another process (scripts/users.py) writes too. Reads see one snapshot.
        with self._lock:
            db = self._connect()
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield db
                db.execute('COMMIT')
            except BaseException:
                if db.in_transaction:
                    db.execute('ROLLBACK')
                raise

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @staticmethod
    def _next_seq(db: sqlite3.Connection) -> int:
        db.execute("UPDATE meta SET value = value + 1 WHERE key = 'seq'")
        return db.execute("SELECT value FROM meta WHERE key = 'seq'").fetchone()[0]

    @staticmethod
    def _usage(db: sqlite3.Connection, user_id: int) -> int:
        chats = db.execute(f'SELECT COALESCE(SUM({stored_bytes("data")}), 0) FROM chats WHERE user_id = ?',
                           (user_id,)).fetchone()[0]
        files = db.execute('SELECT COALESCE(SUM(size), 0) FROM files WHERE user_id = ?', (user_id,)).fetchone()[0]
        return chats + files

    def _within_quota(self, user_id: int, write: Callable[[], object]):
        """Run a write that raises _OverQuota when it does not fit; collect garbage once before giving up."""
        try:
            return write()
        except _OverQuota:
            pass
        if self.collect_garbage(user_id):
            try:
                return write()
            except _OverQuota:
                pass
        raise quota_exceeded()

    # Users

    def local_user(self) -> int:
        """The implicit user of local mode, created on first use."""
        with self._transaction() as db:
            row = db.execute('SELECT id FROM users WHERE name = ?', (LOCAL_USER,)).fetchone()
        if row:
            return row[0]
        with self._transaction(write=True) as db:
            db.execute('INSERT OR IGNORE INTO users (name, password_hash, created_at) VALUES (?, ?, ?)',
                       (LOCAL_USER, LOCAL_HASH, self.now()))
            return db.execute('SELECT id FROM users WHERE name = ?', (LOCAL_USER,)).fetchone()[0]

    def find_user(self, name: str) -> tuple[int, str, str] | None:
        """(id, name as stored, password hash); names compare without regard to ASCII case."""
        with self._transaction() as db:
            return db.execute('SELECT id, name, password_hash FROM users WHERE name = ?', (name,)).fetchone()

    def add_user(self, name: str, password_hash: str) -> int | None:
        """The new user's id, or None when the name is taken."""
        with self._transaction(write=True) as db:
            if db.execute('SELECT 1 FROM users WHERE name = ?', (name,)).fetchone():
                return None
            return db.execute('INSERT INTO users (name, password_hash, created_at) VALUES (?, ?, ?)',
                              (name, password_hash, self.now())).lastrowid

    def set_password(self, name: str, password_hash: str) -> bool:
        with self._transaction(write=True) as db:
            row = db.execute('SELECT id FROM users WHERE name = ?', (name,)).fetchone()
            if row is None:
                return False
            db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, row[0]))
            db.execute('DELETE FROM sessions WHERE user_id = ?', (row[0],))  # every device signs in again
            return True

    def remove_user(self, name: str) -> bool:
        """Delete the user with all their sessions, chats, settings and files."""
        with self._transaction(write=True) as db:
            row = db.execute('SELECT id FROM users WHERE name = ?', (name,)).fetchone()
            if row is None:
                return False
            files = [file_id for (file_id,) in db.execute('SELECT id FROM files WHERE user_id = ?', (row[0],))]
            db.execute('DELETE FROM users WHERE id = ?', (row[0],))
        for file_id in files:
            self.file_path(file_id).unlink(missing_ok=True)
        return True

    def users(self) -> list[dict]:
        with self._transaction() as db:
            rows = db.execute(f'''
                SELECT u.name, u.password_hash, u.created_at,
                       (SELECT COUNT(*) FROM chats c WHERE c.user_id = u.id AND c.deleted = 0),
                       (SELECT COALESCE(SUM({stored_bytes("c.data")}), 0) FROM chats c WHERE c.user_id = u.id)
                       + (SELECT COALESCE(SUM(f.size), 0) FROM files f WHERE f.user_id = u.id)
                FROM users u ORDER BY u.created_at, u.id''').fetchall()
        return [{'name': name, 'can_sign_in': not password_hash.startswith('!'), 'created_at': created,
                 'chats': chats, 'bytes': used} for name, password_hash, created, chats, used in rows]

    def adopt_local(self, user_id: int) -> tuple[int, int]:
        """Hand the local-mode user's chats, settings and uploads to an account; returns (moved, kept).

        A chat the account already has is kept as it is there, and the local copy stays too. Each moved
        chat leaves a tombstone behind, so a browser still caching it in local mode drops it instead of
        saving it back.
        """
        with self._transaction(write=True) as db:
            row = db.execute('SELECT id FROM users WHERE name = ?', (LOCAL_USER,)).fetchone()
            if row is None or row[0] == user_id:
                return 0, 0
            local, moved, kept, now = row[0], 0, 0, self.now()
            for (chat_id,) in db.execute('SELECT chat_id FROM chats WHERE user_id = ? AND deleted = 0',
                                         (local,)).fetchall():
                if db.execute('SELECT 1 FROM chats WHERE user_id = ? AND chat_id = ?', (user_id, chat_id)).fetchone():
                    kept += 1
                    continue
                # New sequence numbers, so the account's devices and local-mode browsers both notice.
                db.execute('UPDATE chats SET user_id = ?, seq = ? WHERE user_id = ? AND chat_id = ?',
                           (user_id, self._next_seq(db), local, chat_id))
                db.execute('INSERT INTO chats (user_id, chat_id, rev, seq, deleted, data, updated_at) '
                           'VALUES (?, ?, 0, ?, 1, NULL, ?)', (local, chat_id, self._next_seq(db), now))
                moved += 1
            db.execute('UPDATE files SET user_id = ? WHERE user_id = ?', (user_id, local))
            if not db.execute('SELECT 1 FROM settings WHERE user_id = ?', (user_id,)).fetchone():
                db.execute('UPDATE settings SET user_id = ? WHERE user_id = ?', (user_id, local))
        return moved, kept

    def account_count(self) -> int:
        """Users who can sign in, which excludes the implicit local user."""
        with self._transaction() as db:
            return db.execute("SELECT COUNT(*) FROM users WHERE substr(password_hash, 1, 1) != '!'").fetchone()[0]

    # Sessions: only a digest of each token is stored, so a copy of the database cannot sign anyone in.

    def add_session(self, user_id: int, token_hash: str, expires_at: int) -> None:
        now = self.now()
        with self._transaction(write=True) as db:
            db.execute('DELETE FROM sessions WHERE expires_at <= ?', (now,))
            db.execute('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
                       (token_hash, user_id, now, expires_at))

    def session_user(self, token_hash: str) -> tuple[int, str] | None:
        """(user id, name) for a live session of a user who may sign in."""
        with self._transaction() as db:
            row = db.execute('''SELECT s.user_id, u.name, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id
                                WHERE s.token_hash = ? AND substr(u.password_hash, 1, 1) != '!' ''',
                             (token_hash,)).fetchone()
        if row is None:
            return None
        if row[2] <= self.now():
            self.remove_session(token_hash)
            return None
        return row[0], row[1]

    def remove_session(self, token_hash: str) -> None:
        with self._transaction(write=True) as db:
            db.execute('DELETE FROM sessions WHERE token_hash = ?', (token_hash,))

    # Chats are stored as the JSON text the client sent; the server reads only their id and rev.

    def changes(self, user_id: int, since: int) -> tuple[int, list[str], list[str]]:
        """(current sequence, chats changed after `since` as JSON text, ids deleted after `since`)."""
        with self._transaction() as db:
            seq = db.execute("SELECT value FROM meta WHERE key = 'seq'").fetchone()[0]
            rows = db.execute('SELECT chat_id, deleted, data FROM chats WHERE user_id = ? AND seq > ? ORDER BY seq',
                              (user_id, since)).fetchall()
        return seq, [data for _, deleted, data in rows if not deleted], [chat for chat, deleted, _ in rows if deleted]

    def put_chat(self, user_id: int, chat_id: str, rev: int, data: str) -> tuple[int, int]:
        """Store a revision unless the stored one is newer; returns (rev, seq).

        Raises StaleChat when the stored revision is newer and chat_deleted once the chat is deleted.
        """
        stored = self._within_quota(user_id, lambda: self._put_chat(user_id, chat_id, rev, data))
        self._fruitless.pop(user_id, None)  # the new revision may have dropped an image
        return stored

    def _put_chat(self, user_id: int, chat_id: str, rev: int, data: str) -> tuple[int, int]:
        with self._transaction(write=True) as db:
            row = db.execute(f'SELECT rev, seq, deleted, data, {stored_bytes("data")} FROM chats '
                             'WHERE user_id = ? AND chat_id = ?', (user_id, chat_id)).fetchone()
            old_size = 0
            if row:
                stored_rev, stored_seq, deleted, stored, old_size = row
                if deleted:  # deleting always wins, whatever revision another device still holds
                    raise ServiceError('chat_deleted', 'This chat was deleted', 410)
                if stored_rev > rev:
                    raise StaleChat(stored)
                if stored_rev == rev:
                    return rev, stored_seq
            size = len(data.encode())
            # A write that does not grow the chat is always allowed, so a full account can still trim.
            if size > old_size and self._usage(db, user_id) - old_size + size > self.quota_bytes:
                raise _OverQuota
            seq = self._next_seq(db)
            db.execute('''INSERT INTO chats (user_id, chat_id, rev, seq, deleted, data, updated_at)
                          VALUES (?, ?, ?, ?, 0, ?, ?)
                          ON CONFLICT (user_id, chat_id) DO UPDATE SET rev = excluded.rev, seq = excluded.seq,
                              data = excluded.data, updated_at = excluded.updated_at''',
                       (user_id, chat_id, rev, seq, data, self.now()))
            return rev, seq

    def delete_chat(self, user_id: int, chat_id: str) -> None:
        """Leave a tombstone, even for a chat never stored, so devices that still hold it drop it."""
        with self._transaction(write=True) as db:
            row = db.execute('SELECT deleted FROM chats WHERE user_id = ? AND chat_id = ?',
                             (user_id, chat_id)).fetchone()
            if row and row[0]:
                return
            db.execute('''INSERT INTO chats (user_id, chat_id, rev, seq, deleted, data, updated_at)
                          VALUES (?, ?, 0, ?, 1, NULL, ?)
                          ON CONFLICT (user_id, chat_id) DO UPDATE SET seq = excluded.seq, deleted = 1,
                              data = NULL, updated_at = excluded.updated_at''',
                       (user_id, chat_id, self._next_seq(db), self.now()))
        self._fruitless.pop(user_id, None)  # its images may be garbage now

    # Settings

    def settings(self, user_id: int) -> str | None:
        with self._transaction() as db:
            row = db.execute('SELECT data FROM settings WHERE user_id = ?', (user_id,)).fetchone()
        return row[0] if row else None

    def put_settings(self, user_id: int, data: str) -> None:
        with self._transaction(write=True) as db:
            db.execute('''INSERT INTO settings (user_id, data, updated_at) VALUES (?, ?, ?)
                          ON CONFLICT (user_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at''',
                       (user_id, data, self.now()))

    # Files: content lives at files/<first two characters of the id>/<id>; ids are made here, never taken
    # from a caller as a path.

    def file_path(self, file_id: str) -> Path:
        return self.data_dir / 'files' / file_id[:2] / file_id

    def add_file(self, user_id: int, mime: str, data: bytes) -> str:
        """Store an already validated image and return its new id."""
        with self._lock:
            self._connect()  # creates the data directory with its permissions first
        file_id = secrets.token_urlsafe(18)
        path = self.file_path(file_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(file_id + '.partial')
        try:
            partial.write_bytes(data)
            partial.replace(path)  # the file is complete before any row can point at it
            self._within_quota(user_id, lambda: self._insert_file(user_id, file_id, mime, len(data)))
        except BaseException:
            partial.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise
        return file_id

    def _insert_file(self, user_id: int, file_id: str, mime: str, size: int) -> None:
        with self._transaction(write=True) as db:
            if self._usage(db, user_id) + size > self.quota_bytes:
                raise _OverQuota
            db.execute('INSERT INTO files (id, user_id, mime, size, created_at) VALUES (?, ?, ?, ?, ?)',
                       (file_id, user_id, mime, size, self.now()))

    def find_file(self, user_id: int, file_id: str) -> tuple[str, Path] | None:
        """(MIME type, path) of a file owned by this user."""
        if not FILE_ID.fullmatch(file_id):
            return None
        with self._transaction() as db:
            row = db.execute('SELECT mime FROM files WHERE id = ? AND user_id = ?', (file_id, user_id)).fetchone()
        path = self.file_path(file_id)
        return (row[0], path) if row and path.is_file() else None

    def read_file(self, user_id: int, file_id: str) -> tuple[str, bytes] | None:
        found = self.find_file(user_id, file_id)
        if found is None:
            return None
        try:
            return found[0], found[1].read_bytes()
        except FileNotFoundError:
            return None

    def collect_garbage(self, user_id: int) -> int:
        """Delete this user's older uploads that no stored chat refers to; returns the bytes freed."""
        now = self.now()
        if now - self._fruitless.get(user_id, -RECOLLECT_SECONDS) < RECOLLECT_SECONDS:
            return 0
        with self._transaction(write=True) as db:
            old = dict(db.execute('SELECT id, size FROM files WHERE user_id = ? AND created_at < ?',
                                  (user_id, now - FILE_GRACE_SECONDS)).fetchall())
            used = set()
            if old:
                for (data,) in db.execute('SELECT data FROM chats WHERE user_id = ? AND deleted = 0', (user_id,)):
                    used.update(FILE_REFERENCE.findall(data))
            garbage = [file_id for file_id in old if file_id not in used]
            db.executemany('DELETE FROM files WHERE id = ?', [(file_id,) for file_id in garbage])
        for file_id in garbage:
            self.file_path(file_id).unlink(missing_ok=True)
        if garbage:
            self._fruitless.pop(user_id, None)
        else:
            self._fruitless[user_id] = now
        return sum(old[file_id] for file_id in garbage)

    def usage(self, user_id: int) -> int:
        """Bytes this user stores: chat JSON plus files."""
        with self._transaction() as db:
            return self._usage(db, user_id)
