import base64
import io
import json
import threading
import anyio
import pytest
from fastapi.testclient import TestClient
from tests.test_api import client, events_from
from tests.test_auth import accounts, sign_in
from tests.test_chat_service import Generator


def chat(chat_id='c1', rev=1, **fields):
    return {'id': chat_id, 'rev': rev, 'title': '새 대화', 'messages': [], **fields}


def image(fmt='PNG', size=(16, 12)):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, 'red').save(buffer, fmt)
    return buffer.getvalue()


def upload(c, data, mime='image/png'):
    return c.post('/api/files', content=data, headers={'Content-Type': mime})


def test_chats_sync_by_revision_sequence_and_tombstone():
    c, _, _ = client()
    with c:
        body = chat(messages=[{'id': 'm1', 'role': 'user', 'content': '안녕하세요 \U0001f600',
                               'nested': {'a': [1, 2.5, None]}}])
        r = c.put('/api/chats/c1', json=body)
        assert r.status_code == 200
        first = r.json()['seq']
        assert r.json() == {'rev': 1, 'seq': first}
        state = c.get('/api/chats').json()
        assert state == {'seq': first, 'chats': [body], 'deleted': [], 'live': []}
        assert c.put('/api/chats/c1', json=chat(title='same revision')).json() == {'rev': 1, 'seq': first}
        r = c.put('/api/chats/c1', json=chat(rev=0, title='older'))
        assert r.status_code == 409 and r.json()['error']['code'] == 'stale_chat'
        assert r.json()['chat'] == body
        second = c.put('/api/chats/c2', json=chat('c2', 1_760_000_000_000)).json()['seq']
        assert c.get('/api/chats', params={'since': first}).json()['chats'] == [chat('c2', 1_760_000_000_000)]
        assert c.delete('/api/chats/c1').status_code == 204
        state = c.get('/api/chats', params={'since': second}).json()
        assert state['chats'] == [] and state['deleted'] == ['c1'] and state['seq'] > second
        r = c.put('/api/chats/c1', json=chat(rev=99))
        assert r.status_code == 410 and r.json()['error']['code'] == 'chat_deleted'
        assert c.delete('/api/chats/never-saved').status_code == 204
        assert c.get('/api/chats', params={'since': state['seq']}).json()['deleted'] == ['never-saved']
        assert c.get('/api/chats', params={'since': -1}).status_code == 422


@pytest.mark.parametrize('body', [
    chat('other'), {'rev': 1}, chat(rev=None), chat(rev=1.5), chat(rev=-1), chat(rev=True), chat(rev='1'),
    chat(rev=2 ** 53), [chat()], 'not json', '{"id":"c1","rev":1,"x":NaN}', '{"id":"c1","rev":1,"x":Infinity}',
    '\N{BYTE ORDER MARK}{"id":"c1","rev":1}', b'{"id":"c1","rev":1,"t":"\xff"}'])
def test_chat_bodies_must_be_the_chat_with_an_integer_revision(body):
    c, _, _ = client()
    with c:
        content = body if isinstance(body, (str, bytes)) else json.dumps(body)
        r = c.put('/api/chats/c1', content=content, headers={'Content-Type': 'application/json'})
        assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_chat'
        assert c.get('/api/chats').json()['chats'] == []


@pytest.mark.parametrize('chat_id', ['a b', 'x' * 65, 'a.b', '한글'])
def test_chat_ids_are_limited_to_url_safe_characters(chat_id):
    c, _, _ = client()
    with c:
        assert c.put(f'/api/chats/{chat_id}', json=chat(chat_id)).status_code == 422
        assert c.delete(f'/api/chats/{chat_id}').status_code == 422


def test_chats_over_four_megabytes_are_refused():
    c, _, _ = client()
    with c:
        big = chat(messages=[{'content': 'x' * (4 * 1024 * 1024)}])
        r = c.put('/api/chats/c1', json=big)
        assert r.status_code == 413 and r.json()['error']['code'] == 'chat_too_large'


def test_storage_quota_covers_chats_and_images():
    c, _, _ = client(user_storage_mb=1)
    with c:
        assert c.put('/api/chats/c1', json=chat(messages=[{'content': 'x' * 700_000}])).status_code == 200
        r = c.put('/api/chats/c2', json=chat('c2', messages=[{'content': 'x' * 400_000}]))
        assert r.status_code == 413 and r.json()['error']['code'] == 'storage_quota'
        picture = image_with_payload(400_000)
        r = upload(c, picture)
        assert r.status_code == 413 and r.json()['error']['code'] == 'storage_quota'
        assert c.put('/api/chats/c1', json=chat(rev=2)).status_code == 200  # shrinking frees the space
        assert upload(c, picture).status_code == 201


def image_with_payload(size: int) -> bytes:
    """A valid PNG padded by a text chunk right after its header, so its size does not depend on compression."""
    import struct
    import zlib
    png = image()
    chunk = b'tEXt' + b'pad\x00' + b'x' * size
    return png[:33] + struct.pack('>I', len(chunk) - 4) + chunk + struct.pack('>I', zlib.crc32(chunk)) + png[33:]


def test_settings_are_stored_per_user_and_last_write_wins():
    c, _, _ = accounts('alice', 'bob')
    with c:
        sign_in(c)
        assert c.get('/api/settings').json() == {'settings': None}
        assert c.put('/api/settings', json={'systemPrompt': '한국어로', 'temperature': 0.2}).status_code == 204
        assert c.put('/api/settings', json={'systemPrompt': '짧게', 'temperature': 0.5}).status_code == 204
        assert c.get('/api/settings').json() == {'settings': {'systemPrompt': '짧게', 'temperature': 0.5}}
        r = c.put('/api/settings', content=json.dumps({'x': 'y' * 33 * 1024}),
                  headers={'Content-Type': 'application/json'})
        assert r.status_code == 413 and r.json()['error']['code'] == 'settings_too_large'
        r = c.put('/api/settings', json=[1, 2])
        assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_settings'
        c.cookies.clear()
        sign_in(c, 'bob')
        assert c.get('/api/settings').json() == {'settings': None}


@pytest.mark.parametrize('fmt,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp'),
                                      ('GIF', 'image/gif')])
def test_uploads_are_typed_by_their_bytes_not_their_header(fmt, mime):
    c, _, _ = client()
    with c:
        data = image(fmt)
        r = upload(c, data, 'image/gif' if mime != 'image/gif' else 'image/png')
        assert r.status_code == 201
        stored = r.json()
        assert stored['mime'] == mime and stored['size'] == len(data) and stored['url'] == '/api/files/' + stored['id']
        r = c.get(stored['url'])
        assert r.status_code == 200 and r.content == data
        assert r.headers['content-type'] == mime and r.headers['x-content-type-options'] == 'nosniff'
        assert r.headers['cache-control'] == 'private, max-age=31536000, immutable'


def test_multi_picture_camera_jpegs_are_stored_as_jpeg():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buffer, 'MPO', save_all=True, append_images=[Image.new('RGB', (8, 8), 'blue')])
    c, _, _ = client()
    with c:
        assert upload(c, buffer.getvalue(), 'image/jpeg').json()['mime'] == 'image/jpeg'


def test_uploads_that_are_not_allowed_images_are_refused():
    from PIL import Image
    c, _, _ = client()
    with c:
        for mime in ['text/plain', 'image/svg+xml', 'application/octet-stream', '']:
            r = c.post('/api/files', content=image(), headers={'Content-Type': mime})
            assert r.status_code == 415 and r.json()['error']['code'] == 'unsupported_media_type'
        r = upload(c, b'x' * (8 * 1024 * 1024 + 1))
        assert r.status_code == 413 and r.json()['error']['code'] == 'file_too_large'
        bmp = io.BytesIO()
        Image.new('RGB', (8, 8)).save(bmp, 'BMP')
        for data in [b'not an image', bmp.getvalue(), image()[:60], b'']:
            r = upload(c, data)
            assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_image'
        assert upload(c, image(), 'IMAGE/PNG; charset=binary').status_code == 201


def test_files_are_visible_only_to_their_owner():
    c, g, _ = accounts('alice', 'bob')
    with c:
        sign_in(c)
        url = upload(c, image()).json()['url']
        assert c.get(url).status_code == 200
        c.cookies.clear()
        sign_in(c, 'bob')
        r = c.get(url)
        assert r.status_code == 404 and r.json()['error']['code'] == 'not_found'
        assert c.get('/api/files/..%2Fchat.db').status_code == 404
        message = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                               {'type': 'text', 'text': '무엇이 보이나요?'}]}
        r = c.post('/v1/chat/completions', json={'messages': [message], 'use_pubmed': False})
        assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_image'
        assert g.messages == []


def test_completions_receive_the_owners_upload_as_a_data_url():
    c, g, _ = client(['빨간 사각형'])
    with c:
        data = image()
        url = upload(c, data, 'image/webp').json()['url']
        message = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                               {'type': 'text', 'text': '무엇이 보이나요?'}]}
        r = c.post('/v1/chat/completions', json={'messages': [message], 'use_pubmed': False, 'chat_id': 'c1'})
        assert r.status_code == 200
        part = g.messages[0][-1]['content'][0]
        assert part['image_url']['url'] == 'data:image/png;base64,' + base64.b64encode(data).decode()
        missing = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': '/api/files/nope'}}]}
        r = c.post('/v1/chat/completions', json={'messages': [missing], 'use_pubmed': False})
        assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_image'
        assert len(g.messages) == 1


def test_chat_ids_in_completions_are_validated():
    c, _, _ = client()
    with c:
        for chat_id in ['a b', 'x' * 65, '']:
            r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}],
                                                     'use_pubmed': False, 'chat_id': chat_id})
            assert r.status_code == 422


class Waiting(Generator):
    """Holds every generation until released, so tests can look at the server meanwhile."""
    def __init__(self, outputs):
        super().__init__(outputs)
        self.started = threading.Semaphore(0)
        self.release = threading.Event()

    async def generate(self, *args, **kwargs):
        self.started.release()
        await anyio.to_thread.run_sync(self.release.wait, 10)
        return await super().generate(*args, **kwargs)


def in_background(call):
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault('response', call()))
    thread.start()
    return thread, result


def stream_request(c, chat_id=None):
    return lambda: c.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': 'hi'}], 'use_pubmed': False, 'stream': True,
        **({'chat_id': chat_id} if chat_id else {})})


def test_a_streaming_chat_is_live_until_its_answer_ends():
    g = Waiting(['hello'])
    c, _, _ = client(generator=g)
    with c:
        thread, result = in_background(stream_request(c, 'chat-1'))
        assert g.started.acquire(timeout=10)
        assert c.get('/api/chats').json()['live'] == ['chat-1']
        g.release.set()
        thread.join(10)
        assert events_from(result['response'])[-1]['type'] == 'done'
        assert c.get('/api/chats').json()['live'] == []


def test_only_two_answers_per_user_run_at_once():
    g = Waiting(['one', 'two', 'three', 'four'])
    c, _, _ = accounts('alice', 'bob', generator=g)
    with c:
        sign_in(c)
        running = [in_background(stream_request(c, f'chat-{n}')) for n in range(2)]
        for _ in running:
            assert g.started.acquire(timeout=10)
        r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}], 'use_pubmed': False})
        assert r.status_code == 429 and r.json()['error']['code'] == 'busy'
        assert c.get('/api/chats').json()['live'] == ['chat-0', 'chat-1']
        bob = TestClient(c.app)
        sign_in(bob, 'bob')
        thread, result = in_background(stream_request(bob))  # another user is not held back
        assert g.started.acquire(timeout=10)
        g.release.set()
        for thread, result in [*running, (thread, result)]:
            thread.join(10)
            assert events_from(result['response'])[-1]['type'] == 'done'
        r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}], 'use_pubmed': False})
        assert r.status_code == 200 and c.get('/api/chats').json()['live'] == []


def test_failed_requests_give_their_slot_back():
    c, _, _ = client(['a', 'b', 'c'])
    with c:
        missing = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': '/api/files/nope'}}]}
        for _ in range(3):
            assert c.post('/v1/chat/completions', json={'messages': [missing], 'stream': True}).status_code == 422
        r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}], 'use_pubmed': False})
        assert r.status_code == 200


def test_idle_streams_send_heartbeat_comments(monkeypatch):
    import app.main
    monkeypatch.setattr(app.main, 'HEARTBEAT_SECONDS', 0.05)
    class Slow(Generator):
        async def generate(self, *args, **kwargs):
            await anyio.sleep(0.3)
            return await super().generate(*args, **kwargs)
    c, _, _ = client(generator=Slow(['hello']))
    with c:
        r = stream_request(c)()
        status, rest = r.text.split('\n\n', 1)
        assert status.startswith('data: ') and rest.startswith(': ping\n\n: ping\n\n')  # while the model "loads"
        assert events_from(r)[-1]['choices'][0]['message']['content'] == 'hello'


def test_a_disconnect_ends_the_stream_and_runs_its_cleanup():
    from app.main import EventStreamResponse
    seen = {}
    async def produce(emit, cancel):
        emit({'type': 'status'})
        await anyio.sleep(30)  # cancelled when the client leaves
    async def main():
        sent = []
        async def receive():
            await anyio.sleep(0.05)
            return {'type': 'http.disconnect'}
        async def send(message):
            sent.append(message)
        response = EventStreamResponse(produce, on_close=lambda: seen.setdefault('closed', len(sent)))
        with anyio.fail_after(5):
            await response({}, receive, send)
        return sent
    sent = anyio.run(main)
    assert sent[0]['type'] == 'http.response.start' and seen['closed'] == len(sent)
    assert b'"status"' in sent[1]['body']
