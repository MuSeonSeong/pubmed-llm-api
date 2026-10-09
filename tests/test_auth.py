import base64
from functools import cache
import time
import pytest
from fastapi.testclient import TestClient
from tests.test_api import client

PASSWORD = 'correct horse'
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")


@cache
def password_hash():
    from app.auth import hash_password
    return hash_password(PASSWORD)


def accounts(*names, **options):
    """An app with sign-in required and an account for each name, all with PASSWORD."""
    c, g, p = client(auth_enabled=True, **options)
    for name in names or ['alice']:
        c.app.state.storage.add_user(name, password_hash())
    return c, g, p


def sign_in(c, name='alice', password=PASSWORD):
    r = c.post('/api/login', json={'name': name, 'password': password})
    assert r.status_code == 200, r.text
    return r


@pytest.fixture
def pages(tmp_path, monkeypatch):
    """Stand-in pages, so these tests do not depend on the web UI's own files."""
    import app.main
    web = tmp_path / 'web'
    web.mkdir()
    (web / 'index.html').write_text('<!doctype html><title>chat page</title>')
    (web / 'login.html').write_text('<!doctype html><title>login page</title>')
    monkeypatch.setattr(app.main, 'WEB_DIR', web)


def test_password_hashes_use_salted_scrypt():
    from app.auth import hash_password, verify_password
    stored = password_hash()
    kind, n, r, p, salt, digest = stored.split('$')
    assert (kind, n, r, p) == ('scrypt', '16384', '8', '1')
    assert len(base64.b64decode(salt)) == 16 and len(base64.b64decode(digest)) == 32
    assert verify_password(PASSWORD, stored) and not verify_password(PASSWORD + '!', stored)
    assert hash_password(PASSWORD) != stored  # a fresh salt every time
    for unusable in ['!local', '!', '', 'scrypt$1$8$1$AA==$AA==', 'scrypt$16384$8$1$!!$AA==', 'bcrypt$x', stored[:-6]]:
        assert not verify_password(PASSWORD, unusable)


@pytest.mark.parametrize('name', ['alice', 'Bob_2', 'a.b-c', '홍길동', 'x' * 32])
def test_valid_names(name):
    from app.auth import check_name
    assert check_name(name) == name


@pytest.mark.parametrize('name', ['', ' ', 'x' * 33, 'a b', 'local', 'LOCAL', 'ㄱㄴ', 'émile', 'a/b', '😀'])
def test_invalid_or_reserved_names(name):
    from app.auth import check_name
    with pytest.raises(ValueError):
        check_name(name)


def test_decomposed_hangul_names_are_composed():
    import unicodedata
    from app.auth import check_name, normalize_name
    decomposed = unicodedata.normalize('NFD', '홍길동')
    assert decomposed != '홍길동' and check_name(decomposed) == '홍길동' == normalize_name(' ' + decomposed)


def test_passwords_need_eight_characters():
    from app.auth import check_password
    with pytest.raises(ValueError):
        check_password('x' * 7)
    check_password('x' * 8)


def test_limiter_counts_failures_per_key_within_the_window():
    from app.auth import LoginLimiter
    now = [1000.0]
    limiter = LoginLimiter(limit=3, window=60, clock=lambda: now[0])
    keys = [('client', '1.2.3.4'), ('name', 'alice')]
    for _ in range(3):
        assert limiter.retry_after(keys) == 0
        limiter.failed(keys)
        now[0] += 10
    assert limiter.retry_after(keys) == 30  # the first failure is forgotten 60 s after it happened
    assert limiter.retry_after([('client', '1.2.3.4')]) == 30
    limiter.forget(('name', 'alice'))
    assert limiter.retry_after([('name', 'alice')]) == 0
    now[0] += 30
    assert limiter.retry_after(keys) == 0


def test_sign_in_sets_a_session_cookie_and_sign_out_ends_it():
    c, _, _ = accounts()
    with c:
        assert c.get('/api/session').json() == {'auth_enabled': True, 'user': None}
        r = sign_in(c, 'ALICE')
        assert r.json() == {'user': {'name': 'alice'}}
        cookie = r.headers['set-cookie']
        assert cookie.startswith('pubmed_session=')
        for flag in ['HttpOnly', 'Path=/', 'SameSite=Lax', f'Max-Age={30 * 86400}']:
            assert flag in cookie
        assert 'secure' not in cookie.lower()  # plain HTTP, as on 127.0.0.1 without a tunnel
        assert c.get('/api/session').json() == {'auth_enabled': True, 'user': {'name': 'alice'}}
        assert c.get('/api/chats').status_code == 200
        token = c.cookies['pubmed_session']
        r = c.post('/api/logout')
        assert r.status_code == 204 and 'Max-Age=0' in r.headers['set-cookie']
        assert c.get('/api/chats').status_code == 401
        # The session row is gone too, so the old token no longer works anywhere.
        c.cookies.set('pubmed_session', token)
        assert c.get('/api/chats').status_code == 401
        assert c.post('/api/logout').status_code == 204  # signing out twice is harmless


def test_cookie_is_secure_over_https():
    c, _, _ = accounts(base_url='https://testserver')
    with c:
        cookie = sign_in(c).headers['set-cookie']
        assert '; Secure' in cookie
        assert c.get('/api/session').json()['user'] == {'name': 'alice'}
        assert '; Secure' in c.post('/api/logout').headers['set-cookie']


def test_wrong_password_and_unknown_name_fail_alike():
    c, _, _ = accounts()
    with c:
        c.app.state.storage.local_user()  # exists after local mode was used, but can never sign in
        replies = [c.post('/api/login', json={'name': name, 'password': password})
                   for name, password in [('alice', 'wrong password'), ('nobody', PASSWORD), ('local', PASSWORD),
                                          ('', PASSWORD)]]
        assert {r.status_code for r in replies} == {401}
        assert len({r.text for r in replies}) == 1 and replies[0].json()['error']['code'] == 'invalid_login'
        assert 'set-cookie' not in replies[0].headers
        r = c.post('/api/login', content='{"name":"alice","password":"\\ud800"}',
                   headers={'Content-Type': 'application/json'})
        assert r.status_code in (401, 422)  # refused, not a server error
        assert c.post('/api/login', json={'name': 'alice'}).status_code == 422


def test_failed_sign_ins_are_throttled_per_address_and_per_name():
    c, _, _ = accounts('alice', 'bob', 'carol', address='10.0.0.1')
    with c:
        for _ in range(10):
            assert c.post('/api/login', json={'name': 'alice', 'password': 'wrong password'}).status_code == 401
        r = c.post('/api/login', json={'name': 'alice', 'password': PASSWORD})
        assert r.status_code == 429 and r.json()['error']['code'] == 'too_many_attempts'
        assert 0 < int(r.headers['retry-after']) <= 15 * 60
        other = TestClient(c.app, client=('10.0.0.2', 50000))
        assert other.post('/api/login', json={'name': 'ALICE', 'password': PASSWORD}).status_code == 429
        assert c.post('/api/login', json={'name': 'bob', 'password': PASSWORD}).status_code == 429
        assert other.post('/api/login', json={'name': 'carol', 'password': PASSWORD}).status_code == 200


def test_expired_or_revoked_sessions_are_refused():
    c, _, _ = accounts()
    with c:
        sign_in(c)
        storage = c.app.state.storage
        storage.clock = lambda: time.time() + 30 * 86400 + 1
        assert c.get('/api/chats').status_code == 401
        storage.clock = time.time
        sign_in(c)
        storage.set_password('alice', password_hash())  # e.g. scripts/users.py passwd
        assert c.get('/api/chats').status_code == 401


PROTECTED = [
    ('GET', '/api/chats', {}, 200),
    ('PUT', '/api/chats/abc', {'json': {'id': 'abc', 'rev': 1}}, 200),
    ('DELETE', '/api/chats/abc', {}, 204),
    ('GET', '/api/settings', {}, 200),
    ('PUT', '/api/settings', {'json': {'theme': 'dark'}}, 204),
    ('POST', '/api/files', {'content': b'not an image', 'headers': {'Content-Type': 'image/png'}}, 422),
    ('GET', '/api/files/abc', {}, 404),
    ('GET', '/v1/models', {}, 200),
    ('POST', '/v1/documents/extract', {'content': b'%PDF-1.4', 'headers': {'Content-Type': 'application/pdf'}}, 422),
    ('POST', '/v1/chat/completions', {'json': {'messages': [{'role': 'user', 'content': 'hi'}], 'use_pubmed': False}},
     200),
]


@pytest.mark.parametrize('method,path,options,status', PROTECTED)
def test_every_api_route_needs_a_session_with_accounts(method, path, options, status):
    c, g, _ = accounts()
    with c:
        for token in [None, 'forged']:
            if token:
                c.cookies.set('pubmed_session', token)
            r = c.request(method, path, **options)
            assert r.status_code == 401 and r.json()['error']['code'] == 'unauthorized'
        assert g.messages == []
        c.cookies.clear()
        sign_in(c)
        assert c.request(method, path, **options).status_code == status


def test_a_signed_in_request_looks_its_session_up_once():
    c, _, _ = accounts()
    with c:
        sign_in(c)
        storage = c.app.state.storage
        lookups = []
        original = storage.session_user
        storage.session_user = lambda digest: lookups.append(digest) or original(digest)
        assert c.get('/api/chats').status_code == 200
        assert len(lookups) == 1


@pytest.mark.parametrize('method,path,options,status', PROTECTED)
def test_local_mode_needs_no_sign_in(method, path, options, status):
    c, _, _ = client()
    with c:
        assert c.request(method, path, **options).status_code == status


def test_local_mode_has_a_fixed_user_and_no_sign_in():
    c, _, _ = client()
    with c:
        assert c.get('/api/session').json() == {'auth_enabled': False, 'user': {'name': 'local'}}
        r = c.post('/api/login', json={'name': 'local', 'password': PASSWORD})
        assert r.status_code == 400 and r.json()['error']['code'] == 'auth_disabled'
        assert c.post('/api/logout').status_code == 204


def test_api_docs_are_only_published_in_local_mode():
    c, _, _ = accounts()
    with c:
        for path in ['/docs', '/redoc', '/openapi.json']:
            assert c.get(path).status_code == 404
        sign_in(c)
        for path in ['/docs', '/redoc', '/openapi.json']:
            assert c.get(path).status_code == 404
    c, _, _ = client()
    with c:
        assert c.get('/openapi.json').status_code == 200 and c.get('/docs').status_code == 200


def test_pages_follow_the_sign_in_state(pages):
    c, _, _ = accounts()
    with c:
        r = c.get('/', follow_redirects=False)
        assert r.status_code == 302 and r.headers['location'] == '/login'
        r = c.get('/login')
        assert r.status_code == 200 and 'login page' in r.text
        assert r.headers['content-security-policy'] == CSP and r.headers['cache-control'] == 'no-cache'
        sign_in(c)
        r = c.get('/login', follow_redirects=False)
        assert r.status_code == 302 and r.headers['location'] == '/'
        assert 'chat page' in c.get('/').text
    c, _, _ = client()
    with c:
        r = c.get('/login', follow_redirects=False)
        assert r.status_code == 302 and r.headers['location'] == '/'
        assert c.get('/').status_code == 200


def test_pages_forbid_inline_code_and_framing():
    c, _, _ = client()
    with c:
        r = c.get('/')
        assert r.headers['content-security-policy'] == CSP
        assert r.headers['x-frame-options'] == 'DENY'
        assert r.headers['cache-control'] == 'no-cache'


def test_every_response_carries_security_headers():
    c, _, _ = accounts()
    with c:
        for r in [c.get('/health'), c.get('/api/session'), c.get('/static/style.css'), c.get('/missing'),
                  c.get('/api/chats'), c.get('/', follow_redirects=False)]:
            assert r.headers['x-content-type-options'] == 'nosniff'
            assert r.headers['referrer-policy'] == 'same-origin'
        assert c.get('/static/style.css').headers['cache-control'] == 'no-cache'
        assert c.get('/api/session').headers['cache-control'] == 'no-store'
        assert 'content-security-policy' not in c.get('/health').headers


def test_unexpected_errors_keep_the_error_shape_and_headers():
    c, _, _ = client()
    with c:
        def broken(user_id, since):
            raise RuntimeError('disk on fire')
        c.app.state.storage.changes = broken
        r = c.get('/api/chats')
        assert r.status_code == 500 and r.json()['error']['code'] == 'internal_error'
        assert 'disk on fire' not in r.text and r.headers['x-content-type-options'] == 'nosniff'


def test_health_tells_anonymous_callers_only_that_the_server_is_up():
    c, _, _ = accounts()
    with c:
        assert c.get('/health').json() == {'status': 'ok'}
        sign_in(c)
        assert c.get('/health').json()['inference']['device'] == 'cpu'


def test_requests_from_other_sites_cannot_change_anything():
    c, g, _ = client()
    with c:
        for site in ['cross-site', 'same-site']:
            r = c.put('/api/chats/abc', json={'id': 'abc', 'rev': 1}, headers={'Sec-Fetch-Site': site})
            assert r.status_code == 403 and r.json()['error']['code'] == 'cross_site_request'
            r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}]},
                       headers={'Sec-Fetch-Site': site})
            assert r.status_code == 403
        assert g.messages == [] and c.get('/api/chats').json()['chats'] == []
        assert c.get('/api/chats', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 200
        r = c.put('/api/chats/abc', json={'id': 'abc', 'rev': 1}, headers={'Sec-Fetch-Site': 'same-origin'})
        assert r.status_code == 200
