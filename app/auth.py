"""Accounts mode: scrypt password hashes, sign-in throttling and the session cookie."""
import base64
from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
import math
import re
import secrets
import threading
import time
import unicodedata
import anyio
from fastapi import Request, Response
from app.settings import ServiceError, Settings
from app.storage import LOCAL_USER, Storage

COOKIE = 'pubmed_session'
# Paths under /api/ and /v1/ that answer without a session; everything else there needs one.
PUBLIC_API = frozenset({'/api/session', '/api/login', '/api/logout'})
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1  # 16 MiB and a few tens of milliseconds per hash
SCRYPT_MAXMEM = 64 * 1024 * 1024
# Checked when the name does not exist, so a wrong name costs the same time as a wrong password.
DUMMY_HASH = '$'.join(['scrypt', str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
                       base64.b64encode(bytes(16)).decode(), base64.b64encode(bytes(32)).decode()])
NAME = re.compile(r'[A-Za-z0-9_.\-가-힣]{1,32}')
MIN_PASSWORD, MAX_PASSWORD = 8, 1024
LOGIN_LIMIT, LOGIN_WINDOW = 10, 15 * 60


@dataclass(frozen=True)
class User:
    id: int
    name: str


def normalize_name(name: str) -> str:
    # Some keyboards send Hangul decomposed into jamo; compose it so it matches the stored name.
    return unicodedata.normalize('NFC', name.strip())


def check_name(name: str) -> str:
    """The normalized name for a new account; ValueError with a message for people otherwise."""
    name = normalize_name(name)
    if not NAME.fullmatch(name):
        raise ValueError("이름은 1~32자의 영문, 숫자, 한글과 '_', '.', '-'로만 지어 주세요.")
    if name.lower() == LOCAL_USER:
        raise ValueError(f"'{LOCAL_USER}'은(는) 로컬 모드용으로 예약된 이름입니다.")
    return name


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f'비밀번호는 {MIN_PASSWORD}자 이상이어야 합니다.')
    if len(password) > MAX_PASSWORD:
        raise ValueError(f'비밀번호는 {MAX_PASSWORD}자 이하여야 합니다.')


def _encode(password: str) -> bytes:
    return password.encode('utf-8', 'surrogatepass')  # a JSON body may carry lone surrogates


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(_encode(password), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32,
                            maxmem=SCRYPT_MAXMEM)
    return '$'.join(['scrypt', str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
                     base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def verify_password(password: str, stored: str) -> bool:
    """False for anything but a matching scrypt hash, including '!...' hashes that disable sign-in."""
    try:
        kind, n, r, p, salt, expected = stored.split('$')
        n, r, p = int(n), int(r), int(p)
        salt, expected = base64.b64decode(salt, validate=True), base64.b64decode(expected, validate=True)
        if kind != 'scrypt' or not expected or not (1 <= r <= 64 and 1 <= p <= 16 and 1 < n <= 2 ** 20):
            return False
        digest = hashlib.scrypt(_encode(password), salt=salt, n=n, r=r, p=p, dklen=len(expected),
                                maxmem=SCRYPT_MAXMEM)
    except (ValueError, MemoryError):  # malformed, or parameters beyond the memory limit
        return False
    return hmac.compare_digest(digest, expected)


def token_digest(token: str) -> str:
    """Sessions are stored by digest, so a copy of the database cannot sign anyone in."""
    return hashlib.sha256(token.encode()).hexdigest()


class LoginLimiter:
    """Failed sign-ins per client address and per name; each failure is forgotten after the window."""

    def __init__(self, limit: int = LOGIN_LIMIT, window: float = LOGIN_WINDOW, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._failures: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: tuple[str, str], now: float) -> deque[float]:
        times = self._failures.get(key)
        if times is None:
            return deque()
        while times and times[0] <= now - self.window:
            times.popleft()
        if not times:
            del self._failures[key]
        return times

    def retry_after(self, keys: list[tuple[str, str]]) -> int:
        """Seconds until all of these keys may try again; 0 when they may now."""
        with self._lock:
            now = self.clock()
            wait = 0.0
            for key in keys:
                times = self._recent(key, now)
                if len(times) >= self.limit:
                    wait = max(wait, times[-self.limit] + self.window - now)
            return max(1, math.ceil(wait)) if wait > 0 else 0

    def failed(self, keys: list[tuple[str, str]]) -> None:
        with self._lock:
            now = self.clock()
            if len(self._failures) > 10_000:  # names sprayed by a guesser must not pile up forever
                for key in list(self._failures):
                    self._recent(key, now)
            for key in keys:
                self._failures.setdefault(key, deque(maxlen=self.limit)).append(now)

    def forget(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._failures.pop(key, None)


def unauthorized() -> ServiceError:
    return ServiceError('unauthorized', 'Sign in to use this server', 401)


class Auth:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings, self.storage = settings, storage
        self.limiter = LoginLimiter()
        # Each hash takes 16 MiB; a burst of sign-ins must not take the memory or every worker thread.
        self._hashing = anyio.CapacityLimiter(2)

    async def local_user(self) -> User:
        return User(await anyio.to_thread.run_sync(self.storage.local_user), LOCAL_USER)

    async def session_user(self, request: Request) -> User | None:
        """The user whose session cookie came with this request, if it is still valid."""
        token = request.cookies.get(COOKIE)
        if not token or len(token) > 128:
            return None
        found = await anyio.to_thread.run_sync(self.storage.session_user, token_digest(token))
        return User(*found) if found else None

    async def current_user(self, request: Request) -> User:
        """FastAPI dependency: the local user in local mode, otherwise the signed-in user or 401."""
        if not self.settings.auth_enabled:
            return await self.local_user()
        user = getattr(request.state, 'user', None) or await self.session_user(request)
        if user is None:
            raise unauthorized()
        return user

    async def login(self, name: str, password: str, client: str) -> tuple[User, str]:
        """Check a password and open a session; returns the user and the cookie token."""
        name = normalize_name(name)
        keys = [('client', client), ('name', name.lower())]
        wait = self.limiter.retry_after(keys)
        if wait:
            raise ServiceError('too_many_attempts', 'Too many failed sign-ins; try again later', 429,
                               headers={'Retry-After': str(wait)})
        found = await anyio.to_thread.run_sync(self.storage.find_user, name) if name else None
        stored = found[2] if found and found[2].startswith('scrypt$') else None
        matched = await anyio.to_thread.run_sync(verify_password, password, stored or DUMMY_HASH,
                                                 limiter=self._hashing)
        if not (matched and stored):
            self.limiter.failed(keys)
            raise ServiceError('invalid_login', 'The name or password is incorrect', 401)
        self.limiter.forget(('name', name.lower()))
        token = secrets.token_urlsafe(32)
        expires = self.storage.now() + self.settings.session_days * 86400
        await anyio.to_thread.run_sync(self.storage.add_session, found[0], token_digest(token), expires)
        return User(found[0], found[1]), token

    async def logout(self, request: Request) -> None:
        token = request.cookies.get(COOKIE)
        if self.settings.auth_enabled and token and len(token) <= 128:
            await anyio.to_thread.run_sync(self.storage.remove_session, token_digest(token))

    def set_cookie(self, response: Response, request: Request, token: str) -> None:
        # Secure whenever the browser came over HTTPS; behind Cloudflare Tunnel uvicorn learns that from
        # X-Forwarded-Proto, which it trusts from 127.0.0.1 only.
        response.set_cookie(COOKIE, token, max_age=self.settings.session_days * 86400, path='/',
                            secure=request.url.scheme == 'https', httponly=True, samesite='Lax')

    def clear_cookie(self, response: Response, request: Request) -> None:
        response.delete_cookie(COOKIE, path='/', secure=request.url.scheme == 'https', httponly=True,
                               samesite='Lax')
