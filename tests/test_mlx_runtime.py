"""The MLX sidecar pool with a fake launcher and a fake mlx_vlm.server behind httpx.MockTransport."""
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import anyio
import httpx
import pytest
from app.settings import ServiceError

# The MLX runtime runs on macOS; Windows has neither these sockets nor these permissions.
unix_only = pytest.mark.skipif(os.name == 'nt', reason='Unix sockets and file permissions')
SEARCH = {'type': 'function', 'function': {'name': 'pubmed_search_articles', 'description': 'Search',
                                           'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}


class Process:
    """A sidecar that runs until signalled; a stubborn one ignores SIGTERM."""
    def __init__(self, stubborn=False):
        self.code, self.stubborn, self.signals, self.pid = None, stubborn, [], 4242

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired('sidecar', timeout)
        return self.code

    def signal(self, sig):
        self.signals.append(sig)
        if sig == signal.SIGKILL or not self.stubborn:
            self.code = -sig


class Launcher:
    def __init__(self, make=Process, log=None, exits=False):
        self.make, self.log, self.exits, self.started = make, log, exits, []

    def __call__(self, command, env, log_path):
        if self.log:
            log_path.write_text(self.log)
        process = self.make()
        if self.exits:
            process.code = 1
        self.started.append((command, env, log_path, process))
        return process


class Stream(httpx.AsyncByteStream):
    def __init__(self, lines):
        self.lines, self.closed = lines, False

    async def __aiter__(self):
        for line in self.lines:
            yield (line + '\n\n').encode()

    async def aclose(self):
        self.closed = True


class Silent(Stream):
    """A long prompt still being read: nothing arrives for a long time."""
    async def __aiter__(self):
        await anyio.sleep(30)
        yield b'data: [DONE]\n\n'


def sse(*deltas, finish='stop', tool_calls=None, usage=(30, 5)):
    """What mlx_vlm.server streams: deltas (reasoning under both names), a finish chunk, usage, [DONE]."""
    def chunk(delta, reason=None):
        if 'reasoning' in delta:
            delta = {**delta, 'reasoning_content': delta['reasoning']}
        return 'data: ' + json.dumps({'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'finish_reason': reason, 'delta': {'role': 'assistant', **delta}}]}, ensure_ascii=False)
    lines = [chunk(delta) for delta in deltas]
    lines.append(chunk({'tool_calls': tool_calls}, 'tool_calls') if tool_calls else chunk({}, finish))
    lines.append('data: ' + json.dumps({'choices': [], 'usage': {
        'prompt_tokens': usage[0], 'completion_tokens': usage[1], 'total_tokens': sum(usage),
        'prompt_tokens_details': {'cached_tokens': 0}}}))
    return [*lines, 'data: [DONE]']


class Server:
    """mlx_vlm.server as the pool sees it: /health once the model is loaded, warm-ups, streamed answers.

    Every sidecar has its own key, which tells them apart on the shared mock transport.
    """
    def __init__(self, lines=(), *, refuse_health=0, status=200, detail='', refuse_chat=False, stream=Stream):
        self.lines, self.refuse_health, self.status, self.detail = list(lines), refuse_health, status, detail
        self.refuse_chat, self.stream, self.requests, self.streams, self.pool = refuse_chat, stream, [], [], None

    def __call__(self, request):
        self.requests.append(request)
        sidecar = next((s for s in self.pool.sidecars.values()
                        if request.headers.get('authorization') == f'Bearer {s.key}'), None)
        if sidecar is None:
            return httpx.Response(401, json={'detail': 'Invalid API key'})
        if request.url.path == '/health':
            if self.refuse_health:  # the socket appears only once the model is in memory
                self.refuse_health -= 1
                raise httpx.ConnectError('refused', request=request)
            return httpx.Response(200, json={'status': 'healthy', 'loaded_model': str(sidecar.path)})
        body = json.loads(request.content)
        if not body['stream']:
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': '네'}}]})
        if self.refuse_chat:
            raise httpx.ConnectError('refused', request=request)
        if self.status != 200:
            return httpx.Response(self.status, json={'detail': self.detail})
        stream = self.stream(self.lines)
        self.streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    def bodies(self, stream=True):
        return [json.loads(r.content) for r in self.requests
                if r.url.path == '/v1/chat/completions' and json.loads(r.content)['stream'] is stream]


class Tokenizer:
    """Counts one token per character of a rendered prompt of a fixed size."""
    def __init__(self, size):
        self.size, self.calls = size, []

    def apply_chat_template(self, messages, tools=None, enable_thinking=False, add_generation_prompt=True,
                            tokenize=True):
        assert tokenize is False and add_generation_prompt is True
        self.calls.append({'messages': messages, 'tools': tools, 'enable_thinking': enable_thinking})
        return 'x' * self.size

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {'input_ids': list(range(len(text)))}


def prepared(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        (path / name).write_text('{}')
    (path / 'model.safetensors').write_bytes(b'weights')


def private_dir(path):
    path.mkdir(mode=0o700, parents=True)
    return path


def make_pool(tmp_path, server=None, *, models=('qwen3.5-9b',), launcher=None, prompt=100, prepare=True,
              **options):
    from app.catalog import load_catalog
    from app.mlx_runtime import MLXPool
    from app.settings import Settings
    catalog = load_catalog()
    settings = Settings(models_dir=tmp_path / 'models', data_dir=tmp_path / 'data', models=','.join(models),
                        **options)
    if prepare:
        for model_id in models:
            prepared(settings.models_dir / catalog[model_id].directory)
    server = server or Server()
    pool = MLXPool(settings, [catalog[model_id] for model_id in models], launcher=launcher or Launcher(),
                   transport=httpx.MockTransport(server))
    server.pool, pool.poll_seconds, pool.stop_timeout = pool, 0.01, 0.05
    pool.socket_dir = lambda longest: private_dir(tmp_path / 'sockets')  # nothing listens in tests
    tokenizer = Tokenizer(prompt)
    pool.load_tokenizer = lambda path: tokenizer
    return pool, server, tokenizer


def spec(model_id='qwen3.5-9b', **changes):
    from app.catalog import load_catalog
    return dataclasses.replace(load_catalog()[model_id], **changes)


def ask(pool, messages=None, tools=(), max_tokens=100, temperature=None, model=None, **options):
    messages = messages or [{'role': 'user', 'content': '하늘은 왜 파란가요?'}]
    return asyncio.run(pool.generate(model or spec(), messages, list(tools), max_tokens, temperature, **options))


async def wait_for(condition, timeout=5):
    with anyio.fail_after(timeout):
        while not condition():
            await anyio.sleep(0.01)


def state(pool, model_id='qwen3.5-9b'):
    return pool.status()[model_id]['state']


NINE_FOUR = ('qwen3.5-9b', 'qwen3.5-4b')
IDLE = 11 * 60  # seconds: longer than MLX_IDLE_MINUTES (10 by default)


async def in_app(pool, preload, body):
    """body() while the app runs: the pool owns the starts, starts the models in preload and checks for idle ones."""
    async with anyio.create_task_group() as app:
        await app.start(pool.run, None, preload)
        await body()
        app.cancel_scope.cancel()


def launched(pool, model_id):
    """The processes started for a model, oldest first."""
    path = str(pool.sidecars[model_id].path)
    return [process for command, _, _, process in pool.launcher.started if command[command.index('--model') + 1] == path]


def test_a_sidecar_is_ready_once_health_names_its_model_and_a_warm_up_answered(tmp_path):
    from app.mlx_runtime import HELLO
    pool, server, tokenizer = make_pool(tmp_path, Server(refuse_health=3))
    assert pool.status()['qwen3.5-9b'] == {'state': 'stopped', 'runtime': 'mlx', 'label': 'Qwen3.5 9B', 'error': None}
    asyncio.run(pool.start_all())
    assert state(pool) == 'ready'
    (command, env, log_path, process), = pool.launcher.started
    path = str(tmp_path / 'models' / 'qwen3.5-9b')
    assert command[1:] == ['-m', 'app.mlx_sidecar', '--host', '127.0.0.1', '--model', path,
                           '--log-progress-interval', '0']
    key = env['MLX_VLM_SERVER_API_KEY']
    assert len(key) > 30 and key not in ' '.join(command)  # argv is visible to every local user
    cache = tmp_path / 'data' / 'kv-cache'
    assert {name: env[name] for name in ['APC_ENABLED', 'APC_CHECKPOINT_INTERVAL_TOKENS', 'APC_CHECKPOINT_ENTRIES',
                                         'APC_DISK_ENABLED', 'MLX_VLM_CACHE_HOME', 'HF_HUB_OFFLINE',
                                         'TOKENIZERS_PARALLELISM']} == {
        'APC_ENABLED': '1', 'APC_CHECKPOINT_INTERVAL_TOKENS': '512', 'APC_CHECKPOINT_ENTRIES': '8',
        'APC_DISK_ENABLED': '0', 'MLX_VLM_CACHE_HOME': str(cache), 'HF_HUB_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false'}
    assert env['MLX_SIDECAR_PARENT_PID'] == str(os.getpid())  # the sidecar exits once this process is gone
    assert env['MLX_SIDECAR_SOCKET'] == str(tmp_path / 'sockets' / '0.sock')
    assert {name for name in env if name.startswith('MLX_SIDECAR_')} == {'MLX_SIDECAR_PARENT_PID', 'MLX_SIDECAR_SOCKET'}
    assert log_path == tmp_path / 'data' / 'logs' / 'mlx-qwen3.5-9b.log'
    assert (tmp_path / 'data' / 'kv-cache').stat().st_mode & 0o777 == 0o700
    health = [r for r in server.requests if r.url.path == '/health']
    assert len(health) == 4 and {r.headers['authorization'] for r in health} == {f'Bearer {key}'}
    assert {r.url.host for r in server.requests} == {'mlx'}
    warm, = server.bodies(stream=False)
    assert (warm['model'], warm['max_tokens'], warm['block_cjk'], warm['enable_thinking']) == (path, 1, True, False)
    assert warm['messages'] == HELLO and 'tools' not in warm
    assert tokenizer.calls == []  # the counting tokenizer is loaded, not used


def test_the_disk_tier_of_the_prefix_cache_is_off_unless_kv_cache_gb_is_set(tmp_path):
    pool, _, _ = make_pool(tmp_path)
    asyncio.run(pool.start_all())
    env = pool.launcher.started[0][1]
    assert env['APC_DISK_ENABLED'] == '0' and env['APC_ENABLED'] == '1'  # in memory only
    assert 'APC_DISK_PATH' not in env and 'APC_DISK_MAX_GB' not in env
    pool, _, _ = make_pool(tmp_path / 'disk', models=('qwen3.5-9b', 'qwen3.5-4b'), kv_cache_gb=3)
    asyncio.run(pool.start_all())
    (_, env, _, _), (_, other, _, _) = pool.launcher.started
    assert (env['APC_DISK_ENABLED'], env['APC_DISK_PATH']) == ('1', str(tmp_path / 'disk' / 'data' / 'kv-cache' /
                                                                         'qwen3.5-9b'))
    assert env['APC_DISK_MAX_GB'] == other['APC_DISK_MAX_GB'] == '1.5'  # KV_CACHE_GB is shared by the models


def test_models_start_one_at_a_time_each_on_its_own_socket_with_its_own_key(tmp_path):
    pool, _, _ = make_pool(tmp_path, models=('qwen3.5-9b', 'qwen3.5-4b'))
    asyncio.run(pool.start_all())
    (_, env, _, _), (_, other, _, _) = pool.launcher.started
    assert (env['MLX_SIDECAR_SOCKET'], other['MLX_SIDECAR_SOCKET']) == (
        str(tmp_path / 'sockets' / '0.sock'), str(tmp_path / 'sockets' / '1.sock'))
    assert env['MLX_VLM_SERVER_API_KEY'] != other['MLX_VLM_SERVER_API_KEY']
    assert {s['state'] for s in pool.status().values()} == {'ready'}


@unix_only
def test_socket_directories_are_private_and_short_enough_for_af_unix(tmp_path, monkeypatch):
    import shutil
    import tempfile
    from app.mlx_runtime import SOCKET_PATH_LIMIT, socket_dir
    short = Path(tempfile.mkdtemp(dir='/tmp'))
    try:
        monkeypatch.setattr(tempfile, 'tempdir', str(short))
        path = socket_dir('1.sock')
        assert path.parent == short and path.name.startswith('pubmed-mlx-')
        assert path.stat().st_mode & 0o777 == 0o700  # uvicorn opens the socket itself to everyone
        long = tmp_path / ('x' * 100)
        long.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(long))
        fallback = socket_dir('1.sock')
        try:
            assert fallback.parent == Path('/tmp') and fallback.stat().st_mode & 0o777 == 0o700
            assert len(os.fsencode(fallback / '1.sock')) < SOCKET_PATH_LIMIT
            assert list(long.iterdir()) == []  # the first try is removed
        finally:
            fallback.rmdir()
    finally:
        shutil.rmtree(short)


@unix_only
def test_the_pool_talks_to_its_sidecar_over_the_private_unix_socket(tmp_path):
    """No mock transport: a stand-in server (no model) on the socket the sidecar was told to use."""
    from app.mlx_runtime import socket_dir
    pool, _, _ = make_pool(tmp_path)
    pool.transport, pool.socket_dir = None, socket_dir
    model, seen, hung_up = str(pool.sidecars['qwen3.5-9b'].path), [], []

    async def handle(reader, writer):
        head = (await reader.readuntil(b'\r\n\r\n')).decode()
        length = re.search(r'content-length: *(\d+)', head, re.I)
        body = json.loads(await reader.readexactly(int(length.group(1)))) if length else None
        if body and body['messages'][-1]['content'] == '아주 긴 논문':  # still reading the prompt
            writer.write(b'HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n')
            await writer.drain()
            hung_up.append(await reader.read() == b'')  # until the app closes the connection
            writer.close()
            return
        seen.append((head.split()[1], head.lower(), body))
        kind = 'application/json'
        if head.startswith('GET /health'):
            data = json.dumps({'status': 'healthy', 'loaded_model': model})
        elif not body['stream']:
            data = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': '네'}}]})
        else:
            data, kind = ''.join(line + '\n\n' for line in sse({'content': '안녕하세요'})), 'text/event-stream'
        writer.write(f'HTTP/1.1 200 OK\r\ncontent-type: {kind}\r\nconnection: close\r\n\r\n'.encode() + data.encode())
        await writer.drain()
        writer.close()

    async def main():
        async with anyio.create_task_group() as group:
            group.start_soon(pool.start_all)
            await wait_for(lambda: pool.launcher.started)
            server = await asyncio.start_unix_server(handle, pool.launcher.started[0][1]['MLX_SIDECAR_SOCKET'])
            await wait_for(lambda: state(pool) == 'ready')
            result = await pool.generate(spec(), [{'role': 'user', 'content': '인사해 줘'}], [], 50)
            cancel = threading.Event()
            group.start_soon(lambda: anyio.to_thread.run_sync(lambda: time.sleep(0.2) or cancel.set()))
            stopped = await pool.generate(spec(), [{'role': 'user', 'content': '아주 긴 논문'}], [], 50, cancel=cancel)
            await wait_for(lambda: hung_up)
            server.close()
        return result, stopped
    try:
        (result, stopped) = asyncio.run(main())
        socket = Path(pool.launcher.started[0][1]['MLX_SIDECAR_SOCKET'])
        assert socket.name == '0.sock' and socket.parent.stat().st_mode & 0o777 == 0o700
    finally:
        pool.close()
    assert not socket.parent.exists()  # the run's directory goes with the app
    assert result.raw_text == '안녕하세요'
    assert (stopped.raw_text, stopped.finish_reason, hung_up) == ('', 'cancelled', [True])  # the server sees Stop
    assert [(target, body and body['stream']) for target, _, body in seen][-3:] == [
        ('/health', None), ('/v1/chat/completions', False), ('/v1/chat/completions', True)]
    key = pool.sidecars['qwen3.5-9b'].key
    assert all(f'authorization: bearer {key.lower()}' in head and 'host: mlx' in head for _, head, _ in seen)


def test_a_crashed_sidecar_is_reported_and_restarted_on_the_next_request(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': '네'})))
    asyncio.run(pool.start_all())
    command, env, log_path, process = pool.launcher.started[0]
    log_path.write_text('loading\n\x1b[31mRuntimeError: [metal] out of memory\x1b[0m\n\n')
    process.code = -9
    status = pool.status()['qwen3.5-9b']
    assert status['state'] == 'failed' and status['error'] == 'exited with code -9: RuntimeError: [metal] out of memory'
    assert ask(pool).raw_text == '네'
    assert len(pool.launcher.started) == 2
    assert pool.launcher.started[1][1]['MLX_VLM_SERVER_API_KEY'] != env['MLX_VLM_SERVER_API_KEY']
    assert pool.launcher.started[1][1]['MLX_SIDECAR_SOCKET'] == env['MLX_SIDECAR_SOCKET']  # the same socket
    assert pool.status()['qwen3.5-9b'] == {'state': 'ready', 'runtime': 'mlx', 'label': 'Qwen3.5 9B', 'error': None}


def test_a_sidecar_that_exits_while_starting_fails_with_its_last_log_line(tmp_path):
    log = 'loading weights\nValueError: [load] Missing 4 parameters: language_model.lm_head.weight\n'
    pool, _, _ = make_pool(tmp_path, launcher=Launcher(log=log, exits=True))
    with pytest.raises(ServiceError) as e:
        asyncio.run(pool.ensure('qwen3.5-9b'))
    assert (e.value.code, e.value.status_code) == ('model_unavailable', 503)
    status = pool.status()['qwen3.5-9b']
    assert status['state'] == 'failed' and status['error'].startswith('exited with code 1: ValueError:')
    assert 'Missing 4 parameters' in status['error'] and 'Missing 4 parameters' in e.value.message
    asyncio.run(pool.start_all())  # failures stay in the state; the background start does not raise
    assert len(pool.launcher.started) == 2


def test_unprepared_models_are_missing_until_their_files_appear(tmp_path):
    pool, _, _ = make_pool(tmp_path, prepare=False)
    assert state(pool) == 'missing'
    with pytest.raises(ServiceError) as e:
        asyncio.run(pool.ensure('qwen3.5-9b'))
    assert (e.value.code, e.value.status_code) == ('model_missing', 503) and pool.launcher.started == []
    prepared(tmp_path / 'models' / 'qwen3.5-9b')
    assert state(pool) == 'stopped'


def test_a_sidecar_that_never_becomes_healthy_times_out(tmp_path):
    pool, _, _ = make_pool(tmp_path, Server(refuse_health=10 ** 6))
    pool.ready_timeout = 0.05
    with pytest.raises(ServiceError):
        asyncio.run(pool.ensure('qwen3.5-9b'))
    assert pool.status()['qwen3.5-9b']['error'] == 'not ready after 0.05 s'
    assert pool.launcher.started[0][3].signals == [signal.SIGTERM]


def test_shutdown_terminates_every_sidecar_and_kills_a_stubborn_one(tmp_path):
    made = iter([Process(), Process(stubborn=True)])
    pool, _, _ = make_pool(tmp_path, models=('qwen3.5-9b', 'qwen3.5-4b'), launcher=Launcher(make=lambda: next(made)))
    asyncio.run(pool.start_all())
    gentle, stubborn = (started[3] for started in pool.launcher.started)
    assert (tmp_path / 'sockets').is_dir()
    pool.close()
    assert gentle.signals == [signal.SIGTERM]
    assert stubborn.signals == [signal.SIGTERM, signal.SIGKILL]
    assert {s['state'] for s in pool.status().values()} == {'stopped'}
    assert not (tmp_path / 'sockets').exists()
    with pytest.raises(ServiceError) as e:  # nothing starts again while the app is stopping
        asyncio.run(pool.ensure('qwen3.5-9b'))
    assert e.value.code == 'model_unavailable' and len(pool.launcher.started) == 2


def test_a_start_made_by_a_caller_is_stopped_when_the_caller_is_cancelled(tmp_path):
    pool, _, _ = make_pool(tmp_path, Server(refuse_health=10 ** 6))

    async def main():
        with anyio.move_on_after(0.1):  # without run(), the caller owns the start
            await pool.start_all()
    asyncio.run(main())
    assert state(pool) == 'stopped'
    assert pool.launcher.started[0][3].signals == [signal.SIGTERM]


def test_a_request_that_leaves_does_not_abandon_the_restart_it_waited_for(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': '네'})))

    async def prefix():
        return 'GUIDANCE', [SEARCH]

    async def main():
        async with anyio.create_task_group() as app:
            await app.start(pool.run, prefix, ['qwen3.5-9b'])
            await wait_for(lambda: len(server.bodies(stream=False)) == 2)  # ready, PubMed prompt cached
            pool.launcher.started[0][3].code = -9  # the sidecar crashes
            server.refuse_health = 20  # and its restart takes a while
            async with anyio.create_task_group() as request:
                request.start_soon(pool.generate, spec(), [{'role': 'user', 'content': '안녕'}], [], 50)
                await wait_for(lambda: len(pool.launcher.started) == 2)
                request.cancel_scope.cancel()  # the browser tab closes while the model restarts
            assert state(pool) == 'starting'
            await wait_for(lambda: state(pool) == 'ready')
            await wait_for(lambda: len(server.bodies(stream=False)) == 4)  # warmed up again, PubMed prompt too
            app.cancel_scope.cancel()
    asyncio.run(main())
    crashed, restarted = (started[3] for started in pool.launcher.started)
    assert restarted.signals == [] and server.bodies() == []  # never stopped; the question was never sent


@pytest.mark.parametrize('exits', [False, True])
def test_requests_waiting_for_a_model_share_one_start_and_each_get_its_outcome(tmp_path, exits):
    log = 'ValueError: Model type qwen3_5 not supported.\n'
    pool, _, _ = make_pool(tmp_path, Server(refuse_health=5), launcher=Launcher(log=log, exits=exits))
    outcomes = []

    async def request():
        try:
            outcomes.append(await pool.ensure('qwen3.5-9b'))
        except ServiceError as exc:
            outcomes.append(exc)

    async def main():
        async with anyio.create_task_group() as app:
            await app.start(pool.run, None, [])  # PRELOAD_MODELS=none: the requests start it
            async with anyio.create_task_group() as requests:
                for _ in range(3):
                    requests.start_soon(request)
            app.cancel_scope.cancel()
    asyncio.run(main())
    assert len(pool.launcher.started) == 1 and len(outcomes) == 3
    if exits:
        assert {(e.code, e.status_code) for e in outcomes} == {('model_unavailable', 503)}
        assert all('Model type qwen3_5 not supported' in e.message for e in outcomes)
        assert len({id(e) for e in outcomes}) == 3  # ChatService adds each request's own details to its error
    else:
        assert outcomes == [pool.sidecars['qwen3.5-9b']] * 3


def test_shutdown_cancels_a_start_under_way_and_stops_its_process(tmp_path):
    pool, _, _ = make_pool(tmp_path, Server(refuse_health=10 ** 6))
    errors = []

    async def request():
        try:
            await pool.ensure('qwen3.5-9b')
        except ServiceError as exc:
            errors.append(exc)

    async def main():
        async with anyio.create_task_group() as requests:
            async with anyio.create_task_group() as app:
                await app.start(pool.run, None, ['qwen3.5-9b'])  # preloaded: the start is under way
                await wait_for(lambda: pool.launcher.started)
                requests.start_soon(request)
                await anyio.sleep(0.05)
                app.cancel_scope.cancel()  # the app stops
    asyncio.run(main())
    assert state(pool) == 'stopped' and pool.launcher.started[0][3].signals == [signal.SIGTERM]
    assert [(e.code, e.message) for e in errors] == [('model_unavailable', 'The server is shutting down')]
    with pytest.raises(ServiceError):  # nothing starts again
        asyncio.run(pool.ensure('qwen3.5-9b'))
    assert len(pool.launcher.started) == 1


@pytest.mark.parametrize('preload,sockets', [
    (['qwen3.5-9b'], ['0.sock']),                          # PRELOAD_MODELS=default
    (['qwen3.5-4b', 'qwen3.5-9b'], ['0.sock', '1.sock']),  # all, the default model first
    ([], [])])                                             # none
def test_only_the_preloaded_models_start_with_the_app(tmp_path, preload, sockets):
    pool, _, _ = make_pool(tmp_path, Server(sse({'content': '네'})), models=NINE_FOUR)

    async def body():
        await wait_for(lambda: all(state(pool, model_id) == 'ready' for model_id in preload))
        await anyio.sleep(0.05)
        assert [Path(env['MLX_SIDECAR_SOCKET']).name for _, env, _, _ in pool.launcher.started] == sockets
        assert [state(pool, model_id) for model_id in NINE_FOUR] == [
            'ready' if model_id in preload else 'stopped' for model_id in NINE_FOUR]
        # The others start on their first question.
        assert (await pool.generate(spec('qwen3.5-4b'), [user('안녕')], [], 50)).raw_text == '네'
        assert state(pool, 'qwen3.5-4b') == 'ready'
    asyncio.run(in_app(pool, preload, body))


@pytest.mark.parametrize('preload,stopped', [
    (['qwen3.5-9b'], ['qwen3.5-4b']),                # PRELOAD_MODELS=default
    (['qwen3.5-9b', 'qwen3.5-4b'], []),              # all
    ([], ['qwen3.5-9b', 'qwen3.5-4b'])])             # none
def test_models_that_were_not_preloaded_are_stopped_after_mlx_idle_minutes(tmp_path, caplog, preload, stopped):
    caplog.set_level('INFO', logger='app.mlx')
    pool, _, _ = make_pool(tmp_path, models=NINE_FOUR)
    pool.idle_check_seconds = 0.01

    async def body():
        for model_id in NINE_FOUR:
            await pool.ensure(model_id)
        await anyio.sleep(0.05)  # five checks, but nothing has been idle for ten minutes
        assert {status['state'] for status in pool.status().values()} == {'ready'}
        for sidecar in pool.sidecars.values():
            sidecar.used -= IDLE
        await wait_for(lambda: all(state(pool, model_id) == 'stopped' for model_id in stopped))
        await anyio.sleep(0.05)
        assert [model_id for model_id in NINE_FOUR if state(pool, model_id) == 'stopped'] == stopped
    asyncio.run(in_app(pool, preload, body))
    for model_id in NINE_FOUR:
        process, = launched(pool, model_id)
        assert process.signals == ([signal.SIGTERM] if model_id in stopped else [])
    assert [r.getMessage() for r in caplog.records if 'idle' in r.getMessage()] == [
        f'stopped {model_id} after 11 min idle' for model_id in stopped]


def test_a_question_after_an_idle_stop_starts_the_model_again(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': '네'})), models=NINE_FOUR)
    pool.idle_check_seconds = 0.01
    four = pool.sidecars['qwen3.5-4b']

    async def body():
        await wait_for(lambda: state(pool) == 'ready')  # the default model, preloaded
        assert (await pool.generate(spec('qwen3.5-4b'), [user('안녕')], [], 50)).raw_text == '네'
        key = four.key
        four.used -= IDLE
        await wait_for(lambda: state(pool, 'qwen3.5-4b') == 'stopped')
        assert pool.status() == {
            'qwen3.5-9b': {'state': 'ready', 'runtime': 'mlx', 'label': 'Qwen3.5 9B', 'error': None},
            'qwen3.5-4b': {'state': 'stopped', 'runtime': 'mlx', 'label': 'Qwen3.5 4B', 'error': None}}
        assert (await pool.generate(spec('qwen3.5-4b'), [user('또 안녕')], [], 50)).raw_text == '네'
        assert state(pool, 'qwen3.5-4b') == 'ready' and four.key != key
    asyncio.run(in_app(pool, ['qwen3.5-9b'], body))
    old, new = launched(pool, 'qwen3.5-4b')
    assert old.signals == [signal.SIGTERM] and new.signals == [] and len(server.bodies()) == 2


def test_the_idle_stop_leaves_alone_a_model_that_is_starting_or_answering(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(refuse_health=30, stream=Silent))
    pool.idle_check_seconds = 0.01
    sidecar, cancel = pool.sidecars['qwen3.5-9b'], threading.Event()

    async def ask_long():  # nothing arrives while a long prompt is read
        result = await pool.generate(spec(), [user('이 논문 전체를 요약해 줘')], [], 50, cancel=cancel)
        assert result.finish_reason == 'cancelled'

    async def body():
        async with anyio.create_task_group() as request:  # a question starts the model and leaves meanwhile
            request.start_soon(pool.generate, spec(), [user('안녕')], [], 50)
            await wait_for(lambda: state(pool) == 'starting')
            request.cancel_scope.cancel()
        sidecar.used -= IDLE  # asked for long ago, and the start goes on
        await wait_for(lambda: state(pool) == 'ready')
        await anyio.sleep(0.05)
        assert state(pool) == 'ready'  # idle only from the end of its start
        async with anyio.create_task_group() as request:
            request.start_soon(ask_long)
            await wait_for(lambda: server.streams)
            sidecar.used -= IDLE
            await anyio.sleep(0.05)
            assert state(pool) == 'ready' and sidecar.in_flight == 1
            cancel.set()
        await anyio.sleep(0.05)
        assert state(pool) == 'ready' and sidecar.in_flight == 0  # its last request ended just now
        sidecar.used -= IDLE
        await wait_for(lambda: state(pool) == 'stopped')
    asyncio.run(in_app(pool, [], body))  # PRELOAD_MODELS=none: the default model can be stopped too
    assert pool.launcher.started[0][3].signals == [signal.SIGTERM]


def test_a_question_that_comes_while_the_idle_stop_takes_the_lock_keeps_the_model(tmp_path):
    """Taking a lock lets other tasks run, so the stop looks again once it holds the lock."""
    pool, _, _ = make_pool(tmp_path)
    pool.idle_check_seconds = 0.01
    sidecar = pool.sidecars['qwen3.5-9b']
    question = sidecar.in_use()

    class Arriving:
        async def __aenter__(self):
            question.__enter__()  # what generate() does for a question that comes in just now

        async def __aexit__(self, *exc):
            pass

    async def body():
        await pool.ensure('qwen3.5-9b')
        lock, sidecar.lock = sidecar.lock, Arriving()
        sidecar.used -= IDLE
        await anyio.sleep(0.05)
        assert state(pool) == 'ready' and sidecar.in_flight == 1
        question.__exit__(None, None, None)
        sidecar.lock = lock
        sidecar.used -= IDLE
        await wait_for(lambda: state(pool) == 'stopped')
    asyncio.run(in_app(pool, [], body))


def test_mlx_idle_minutes_0_never_stops_a_model(tmp_path):
    pool, _, _ = make_pool(tmp_path, mlx_idle_minutes=0)
    pool.idle_check_seconds = 0.01

    async def body():
        sidecar = await pool.ensure('qwen3.5-9b')
        sidecar.used -= 24 * 60 * 60
        await anyio.sleep(0.1)
        assert state(pool) == 'ready'
    asyncio.run(in_app(pool, [], body))
    assert pool.launcher.started[0][3].signals == []


def test_a_question_during_an_idle_stop_waits_until_the_old_sidecar_has_exited(tmp_path):
    """The new sidecar listens on the same socket path, which the old one removes as it exits."""
    class Slow(Process):
        """Exits 0.2 s after SIGTERM, like a server finishing its shutdown."""
        def signal(self, sig):
            self.signals.append(sig)
            if sig == signal.SIGKILL:
                self.code = -sig

        def wait(self, timeout=None):
            if self.code is None and signal.SIGTERM in self.signals:
                time.sleep(0.2)
                self.code = -signal.SIGTERM
            return super().wait(timeout)
    pool, _, _ = make_pool(tmp_path, Server(sse({'content': '네'})), launcher=Launcher(make=Slow))
    pool.idle_check_seconds, pool.stop_timeout = 0.01, 1
    launcher, running = pool.launcher, []

    def launch(command, env, log_path):
        running.append([process.poll() is None for *_, process in launcher.started])
        return launcher(command, env, log_path)
    pool.launcher = launch

    async def body():
        sidecar = await pool.ensure('qwen3.5-9b')
        sidecar.used -= IDLE
        await wait_for(lambda: state(pool) == 'stopped')  # its process is still shutting down
        assert (await pool.generate(spec(), [user('안녕')], [], 50)).raw_text == '네'
        assert state(pool) == 'ready'
    asyncio.run(in_app(pool, [], body))
    assert running == [[], [False]]  # never two sidecars on one socket
    old, new = (started[3] for started in launcher.started)
    assert old.signals == [signal.SIGTERM] and new.signals == []


def test_local_mode_caches_the_pubmed_prompt_once_the_model_is_ready(tmp_path):
    from app.mlx_runtime import HELLO
    pool, server, _ = make_pool(tmp_path)

    async def prefix():
        return 'GUIDANCE', [SEARCH]

    async def main():
        async with anyio.create_task_group() as app:
            await app.start(pool.run, prefix, ['qwen3.5-9b'])
            await wait_for(lambda: len(server.bodies(stream=False)) == 2)
            app.cancel_scope.cancel()
    asyncio.run(main())
    plain, pubmed = server.bodies(stream=False)
    assert plain['messages'] == HELLO and 'tools' not in plain
    assert pubmed['messages'] == [{'role': 'system', 'content': 'GUIDANCE'}, *HELLO]
    assert pubmed['tools'] == [SEARCH] and pubmed['tool_parser'] == 'longcat' and pubmed['max_tokens'] == 1


@pytest.mark.parametrize('pubmed', ['slow', 'offline'])
def test_a_slow_or_offline_pubmed_never_holds_the_model_back(tmp_path, caplog, pubmed):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': '네'})))
    pool.prefix_timeout = 0.3
    asked = []

    async def prefix():
        asked.append(time.monotonic())
        if pubmed == 'offline':
            raise ServiceError('pubmed_unavailable', 'offline', 502)
        await anyio.sleep(30)

    async def main():
        async with anyio.create_task_group() as app:
            await app.start(pool.run, prefix, ['qwen3.5-9b'])
            await wait_for(lambda: asked)
            assert state(pool) == 'ready'
            # A question without PubMed is answered at once, whatever PubMed does.
            assert (await pool.generate(spec(), [{'role': 'user', 'content': '안녕'}], [], 50)).raw_text == '네'
            await wait_for(lambda: 'PubMed prompt not cached' in caplog.text)
            app.cancel_scope.cancel()
    asyncio.run(main())
    assert len(server.bodies(stream=False)) == 1  # only the one-token warm-up


def test_answers_become_raw_text_with_streamed_progress(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'Hel'}, {'content': 'lo'})))
    seen = []
    result = ask(pool, on_text=seen.append)
    assert (result.raw_text, result.finish_reason, result.prompt_tokens, result.completion_tokens) == \
        ('Hello', 'stop', 30, 5)
    assert seen == ['', 'Hel', 'Hello']
    body, = server.bodies()
    assert body['model'] == str(tmp_path / 'models' / 'qwen3.5-9b')
    assert body['stream'] is True and body['stream_options'] == {'include_usage': True}
    assert body['messages'] == [{'role': 'user', 'content': '하늘은 왜 파란가요?'}]
    assert body['context_management'] == [] and isinstance(body['seed'], int)
    assert (body['max_tokens'], body['enable_thinking']) == (100, False) and 'thinking_budget' not in body
    assert {key: body[key] for key in ['temperature', 'top_p', 'top_k', 'presence_penalty']} == \
        {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}
    assert 'tools' not in body and 'tool_parser' not in body
    request = [r for r in server.requests if r.url.path == '/v1/chat/completions'][-1]
    assert request.headers['authorization'] == 'Bearer ' + pool.sidecars['qwen3.5-9b'].key


def test_reasoning_comes_back_in_think_tags_and_gets_the_effort_budget(tmp_path):
    from app.chat_service import split_reasoning
    lines = sse({'reasoning': 'Let me'}, {'reasoning': ' add.'}, {'content': '4'}, finish='length')
    pool, server, tokenizer = make_pool(tmp_path, Server(lines))
    seen = []
    result = ask(pool, effort='light', on_text=seen.append)
    assert result.raw_text == '<think>Let me add.</think>4' and result.finish_reason == 'length'
    assert seen == ['', '<think>Let me', '<think>Let me add.', '<think>Let me add.</think>4']
    assert split_reasoning(seen[2]) == ('Let me add.', None)  # still reasoning until the answer starts
    body, = server.bodies()
    assert (body['enable_thinking'], body['thinking_budget'], body['max_tokens']) == (True, 512, 612)
    assert {key: body[key] for key in ['temperature', 'top_p', 'top_k', 'presence_penalty']} == \
        {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 1.5}
    assert tokenizer.calls[-1]['enable_thinking'] is True


def test_the_context_budget_caps_reasoning_then_the_answer(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'ok'})), prompt=30000)
    ask(pool, max_tokens=1024, effort='max')
    body = server.bodies()[-1]
    assert (body['max_tokens'], body['thinking_budget']) == (32768 - 30000, 32768 - 30000 - 1024)
    pool, server, _ = make_pool(tmp_path / 'full', Server(sse({'content': 'ok'})), prompt=31700)
    seen = []
    result = ask(pool, max_tokens=1024, effort='max', on_text=seen.append)  # 44 tokens: too few to reason with
    body = server.bodies()[-1]
    assert (body['max_tokens'], body['enable_thinking'], body['temperature']) == (1024, False, 0.7)
    assert 'thinking_budget' not in body
    # Reasoning was asked for, so the answer still comes after a (now empty) reasoning part.
    assert result.raw_text == '<think></think>ok' and seen == ['', '<think></think>ok']


def test_overflow_is_refused_before_a_sidecar_starts(tmp_path):
    pool, _, _ = make_pool(tmp_path, prompt=32710)
    for strict, max_tokens in [(True, 512), (False, 100)]:  # 58 tokens left
        with pytest.raises(ServiceError) as e:
            ask(pool, max_tokens=max_tokens, strict_budget=strict)
        assert (e.value.code, e.value.status_code) == ('context_overflow', 413)
    assert pool.launcher.started == []
    pool, server, _ = make_pool(tmp_path / 'shrink', Server(sse({'content': 'ok'})), prompt=32700)
    ask(pool, max_tokens=100)  # 68 tokens are left: the answer shrinks instead
    assert server.bodies()[-1]['max_tokens'] == 68


def test_an_explicit_temperature_overrides_only_the_temperature(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'ok'})))
    ask(pool, temperature=0.3)
    body = server.bodies()[-1]
    assert (body['temperature'], body['top_p'], body['top_k']) == (0.3, 0.8, 20)
    ask(pool, temperature=0.0)
    assert server.bodies()[-1]['temperature'] == 0.0


def test_the_tenant_header_is_sent_only_for_accounts(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'ok'})))
    ask(pool, tenant='user-7-1760000000')
    ask(pool)
    chats = [r for r in server.requests if r.url.path == '/v1/chat/completions']
    assert [r.headers.get('x-apc-tenant') for r in chats] == [None, 'user-7-1760000000', None]  # the warm-up is shared


def test_with_accounts_the_sidecars_keep_no_image_features(tmp_path, monkeypatch):
    monkeypatch.setenv('MLX_SIDECAR_PRIVATE', '1')  # the app's own environment does not decide it
    pool, _, _ = make_pool(tmp_path)
    asyncio.run(pool.start_all())
    assert 'MLX_SIDECAR_PRIVATE' not in pool.launcher.started[0][1]
    pool, _, _ = make_pool(tmp_path / 'accounts', auth_enabled=True)
    asyncio.run(pool.start_all())
    assert pool.launcher.started[0][1]['MLX_SIDECAR_PRIVATE'] == '1'


def test_tool_markup_reaches_the_app_untouched(tmp_path):
    markup = '<tool_call>\n<function=pubmed_search_articles>\n<parameter=query>\nmetformin\n</parameter>\n</function>\n</tool_call>'
    pool, server, tokenizer = make_pool(tmp_path, Server(sse({'content': markup})))
    assert ask(pool, tools=[SEARCH]).raw_text == markup
    body = server.bodies()[-1]
    assert body['tools'] == [SEARCH] and body['tool_parser'] == 'longcat'
    assert tokenizer.calls[-1]['tools'] == [SEARCH]


def test_a_tool_call_written_in_the_reasoning_moves_to_the_answer(tmp_path):
    from app.chat_service import parse_tool_calls, split_reasoning
    call = '<tool_call>\n<function=pubmed_search_articles>\n<parameter=query>\nCRISPR\n</parameter>\n</function>\n</tool_call>'
    pool, _, _ = make_pool(tmp_path, Server(sse({'reasoning': 'I should search.\n' + call})))
    result = ask(pool, tools=[SEARCH], effort='medium')
    reasoning, rest = split_reasoning(result.raw_text)
    assert reasoning == 'I should search.' and rest == call
    assert parse_tool_calls(rest, [SEARCH])[0].arguments == {'query': 'CRISPR'}
    pool, _, _ = make_pool(tmp_path / 'both', Server(sse({'reasoning': 'draft ' + call}, {'content': 'Final ' + call})))
    reasoning, rest = split_reasoning(ask(pool, tools=[SEARCH], effort='medium').raw_text)
    assert reasoning == 'draft ' + call and rest == 'Final ' + call  # the answer has its own call: nothing moves


def test_calls_the_server_parsed_are_written_back_as_markup(tmp_path):
    from app.chat_service import parse_tool_calls
    calls = [{'id': 'x', 'type': 'function', 'index': 0,
              'function': {'name': 'pubmed_search_articles', 'arguments': '{"query": "메트포르민"}'}}]
    pool, server, _ = make_pool(tmp_path, Server(sse(tool_calls=calls)))
    result = ask(pool, tools=[SEARCH], model=spec(tool_parsing='server'))
    assert 'tool_parser' not in server.bodies()[-1]
    assert result.finish_reason == 'stop'
    assert parse_tool_calls(result.raw_text, [SEARCH])[0].arguments == {'query': '메트포르민'}


def test_cancelling_closes_the_stream_and_keeps_the_partial_answer(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'Hel'}, {'content': 'lo'}, {'content': ' world'})))
    cancel, seen = threading.Event(), []

    def on_text(text):
        seen.append(text)
        if text:
            cancel.set()  # the browser's stop button
    result = ask(pool, on_text=on_text, cancel=cancel)
    assert (result.raw_text, result.finish_reason) == ('Hel', 'cancelled')
    assert seen == ['', 'Hel'] and server.streams[0].closed
    assert ask(pool, cancel=cancel).finish_reason == 'cancelled'  # already cancelled: nothing is sent
    assert len(server.bodies()) == 1


def test_stop_ends_an_answer_whose_long_prompt_is_still_being_read(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(stream=Silent))
    asyncio.run(pool.start_all())
    cancel, seen = threading.Event(), []

    async def main():
        async def stop():
            await anyio.sleep(0.2)
            cancel.set()  # pressed while the server is still reading a long prompt
        async with anyio.create_task_group() as group:
            group.start_soon(stop)
            started = time.monotonic()
            result = await pool.generate(spec(), [{'role': 'user', 'content': '이 논문 전체를 요약해 줘'}], [], 100,
                                         on_text=seen.append, cancel=cancel)
            return result, time.monotonic() - started
    result, took = asyncio.run(main())
    assert (result.raw_text, result.finish_reason) == ('', 'cancelled') and seen == ['']
    assert took < 2 and server.streams[0].closed  # closing the stream makes the server drop the request


def test_a_prompt_the_server_finds_too_long_is_a_context_overflow(tmp_path):
    detail = 'Request needs 40000 context tokens (39000 prompt + 1000 max generation), but MAX_KV_SIZE is 32768.'
    pool, _, _ = make_pool(tmp_path, Server(status=400, detail=detail))
    with pytest.raises(ServiceError) as e:
        ask(pool)
    assert (e.value.code, e.value.status_code) == ('context_overflow', 413)
    pool, _, _ = make_pool(tmp_path / 'other', Server(status=500, detail='boom'))
    with pytest.raises(ServiceError) as e:
        ask(pool, cancel=threading.Event())  # with Stop being watched, too
    assert (e.value.code, e.value.status_code) == ('generation_failed', 502) and 'boom' in e.value.message


def test_an_error_in_the_middle_of_the_stream_fails_the_generation(tmp_path):
    lines = ['data: ' + json.dumps({'choices': [{'delta': {'content': 'Hi'}}]}), 'data: {"error": "Metal OOM"}']
    pool, _, _ = make_pool(tmp_path, Server(lines))
    with pytest.raises(ServiceError) as e:
        ask(pool)
    assert (e.value.code, e.value.status_code) == ('generation_failed', 502) and 'Metal OOM' in e.value.message


def test_an_unreachable_sidecar_is_restarted_once_then_reported(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(refuse_chat=True))
    with pytest.raises(ServiceError) as e:
        ask(pool, cancel=threading.Event())
    assert (e.value.code, e.value.status_code) == ('model_unavailable', 503)
    assert 'not responding' in e.value.message
    assert len(pool.launcher.started) == 2 and pool.launcher.started[0][3].signals == [signal.SIGTERM]


def test_continuing_asks_for_the_rest_in_a_new_message(tmp_path):
    from app.mlx_runtime import CONTINUE_PROMPT
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': ' 두 번째 문단.'})))
    messages = [{'role': 'user', 'content': '긴 글을 써 줘'}, {'role': 'assistant', 'content': '첫 문단.'}]
    assert ask(pool, messages, tools=[SEARCH], effort='high', continue_final=True).raw_text == ' 두 번째 문단.'
    body = server.bodies()[-1]
    assert body['messages'][-2:] == [{'role': 'assistant', 'content': '첫 문단.'},
                                     {'role': 'user', 'content': CONTINUE_PROMPT.format(tail='첫 문단.')}]
    assert '«첫 문단.»' in body['messages'][-1]['content']  # where the answer stops, quoted
    assert body['enable_thinking'] is False and 'tools' not in body
    assert body['block_cjk'] is True  # decided by the question, not by the instruction


PARTIAL = '고혈압은 뇌졸중의 위험을 높입니다. 약물 치료와 병행하여 생활 습관을 개선하는 것이'


def test_a_continuation_starts_by_repeating_the_answers_last_words_which_are_dropped(tmp_path):
    from app.mlx_runtime import CONTINUE_PROMPT, answer_tail
    pieces = ['생활 습관을 ', '개선하는 것이', ' 가장 효과', '적입니다.']
    pool, server, _ = make_pool(tmp_path, Server(sse(*({'content': piece} for piece in pieces))))
    seen = []
    messages = [user('고혈압 관리법을 알려 줘'), {'role': 'assistant', 'content': PARTIAL}]
    result = ask(pool, messages, continue_final=True, on_text=seen.append)
    assert result.raw_text == ' 가장 효과적입니다.'  # joins as "…것이 가장 효과적입니다."
    assert all('개선하는' not in text for text in seen)  # the repeated words never reach the page
    tail = answer_tail(PARTIAL)
    assert tail == '약물 치료와 병행하여 생활 습관을 개선하는 것이'  # from the start of a word
    assert server.bodies()[-1]['messages'][-1]['content'] == CONTINUE_PROMPT.format(tail=tail)


@pytest.mark.parametrize('reply,continuation', [
    ('것이 고려해 볼 만합니다.', ' 고려해 볼 만합니다.'),   # only the last word repeated
    ('가장 효과적입니다.', '가장 효과적입니다.'),          # nothing repeated: kept as written
    ('이', '이'),                                         # shorter than the words to repeat
])
def test_whatever_the_model_repeats_is_dropped_once(tmp_path, reply, continuation):
    pool, _, _ = make_pool(tmp_path, Server(sse({'content': reply})))
    messages = [user('고혈압 관리법을 알려 줘'), {'role': 'assistant', 'content': PARTIAL}]
    assert ask(pool, messages, continue_final=True).raw_text == continuation


def user(content):
    return {'role': 'user', 'content': content}


def system(content):
    return {'role': 'system', 'content': content}


ANSWER = {'role': 'assistant', 'content': '...'}


@pytest.mark.parametrize('messages,blocked', [
    ([user('하늘은 왜 파란가요?')], True),
    ([user([{'type': 'text', 'text': '이 그림을'}, {'type': 'text', 'text': '설명해 줘'}])], True),
    ([user('한자리 숫자 두 개를 더해 줘')], True),         # everyday words that only start like 한자 or 한문
    ([user('한자릿수 곱셈 문제를 내 줘')], True),
    ([user('한자세로 오래 앉아 있으면 허리가 아파요')], True),
    ([user('한문단으로 요약해 줘')], True),
    ([user('한문장으로 요약해 줘')], True),
    ([user('한문제만 더 풀어 볼게요')], True),
    ([system('한국어로만 답하고 한자나 중국어를 섞지 마세요'), user('당뇨병이란?')], True),  # negated mentions
    ([user('중국어 논문은 빼고 당뇨병 연구를 찾아줘')], True),
    ([user('한자 없이 한글로만 설명해 줘')], True),
    ([system('Answer in Korean. Do not use Chinese characters.'), user('당뇨병이란?')], True),
    ([system('Never write Japanese or Chinese, not even while reasoning.'), user('당뇨병이란?')], True),
    ([user('이 초록을 일본어로 번역해 줘')], False),       # requests for such writing
    ([user('감사합니다를 중국어 간체로 써 줘')], False),
    ([user('당뇨병의 한자 표기를 알려줘')], False),
    ([user('Translate 감사합니다 into Japanese, please')], False),
    ([user('Translate this into Japanese: 오늘 회의는 세 시에 시작합니다')], False),
    ([user('히라가나로 써 줘')], False),
    ([user('중국어 말고 일본어로 번역해 줘')], False),       # one language negated, the other asked for
    ([user('일본어로 쓰지만 발음은 한글로 적어 줘')], False),  # words that only look like a negation
    ([user('중국어로 마라탕 주문하는 법을 알려줘')], False),
    ([user('일본어로 써 줘. 설명은 빼고')], False),         # a negation in the next sentence does not count
    ([user('No. Translate it into Japanese: 감사합니다')], False),
    ([user("Spell 'piano' in Japanese: 피아노")], False),  # "no" negates as a word, not as the end of piano
    ([user('이 초록을 일본어로 번역해 줘'), ANSWER, user('조금 더 자연스럽게 다듬어 줘')], False),  # follow-ups
    ([user('Translate this into Japanese'), ANSWER, user('조금 더 자연스럽게 다듬어 줘')], False),
    ([system('Give the Hanja of Sino-Korean words.'), user('당뇨병이란?')], False),
    ([user('漢字 "愛"는 어떻게 읽나요?')], False),          # questions written with them
    ([user('ひらがな를 알려줘')], False),
    ([user('「愛」는 무슨 뜻이야?'), {'role': 'assistant', 'content': '사랑입니다.'}, user('예문도 들어 줘')], False),
    ([user('Why is the sky blue?')], False),              # no Korean question
    ([user('하늘은 왜 파란가요?'), ANSWER, user('Thanks!')], False),
    # A request stays a request when other words around it are excluded or negated.
    ([user('이 초록을 일본어로 번역하고 표는 제외해 줘')], False),
    ([user('일본어로 제목 빼고 번역해 줘')], False),
    ([user('일본어로 써 줘, 설명은 빼고')], False),
    ([user('일본어로 실수 없이 번역해 줘')], False),
    ([user('일본어로 번역하되 한자는 쓰지 마')], False),
    ([user('중국어로 하되 번체는 쓰지 마')], False),
    ([user('일본어로번역해줘 감사합니다')], False),
    ([user('Not Chinese but Japanese: 감사합니다')], False),
    ([user('Translate everything except the title into Japanese: 오늘 회의는 세 시입니다')], False),
    ([user('Translate without errors into Japanese: 감사합니다')], False),
    # A request whose own verb is negated, and words that only contain a script's name, ask for nothing.
    ([user('한자로 쓰지 마세요. 당뇨병이란?')], True),
    ([system('중국어로 답하지 마세요'), user('당뇨병이란?')], True),
    ([system('Do not answer in Chinese.'), user('당뇨병이란?')], True),
    ([system('Never translate into Chinese, but answer in Korean'), user('당뇨병이란?')], True),
    ([user('공간체험 프로그램이 치매에 효과 있나요?')], True),
    ([user('시간체크로 기록해 줘')], True),
    ([user('한자녀 가정 지원 정책은?')], True),
    ([user('한자루의 연필을 샀어요')], True),
    # Attached files are not read for request words: an English abstract about Chinese adults asks for nothing.
    ([user('이 논문을 요약해 줘\n\n[첨부 파일: abstract.txt]\n```\nWe followed 5,000 Chinese adults; translated '
           'into Chinese.\n```'), ANSWER, user('결과를 더 자세히 설명해 줘')], True),
    ([], False)])
def test_chinese_and_japanese_are_blocked_only_where_nobody_wants_them(messages, blocked):
    from app.mlx_runtime import blocks_cjk
    assert blocks_cjk(messages) is blocked


def test_a_huge_message_is_judged_quickly():
    import time
    from app.mlx_runtime import blocks_cjk
    started = time.perf_counter()
    assert blocks_cjk([user('가 ' + 'not Chinese ' * 5_000_000)]) is True  # 60 MB: only its ends are read
    assert time.perf_counter() - started < 1


def test_the_request_carries_block_cjk_as_decided_by_the_conversation(tmp_path):
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'ok'})))
    ask(pool, [user('이전 질문'), {'role': 'assistant', 'content': '답'}, user([{'type': 'text', 'text': '설명해 줘'}])])
    assert server.bodies()[-1]['block_cjk'] is True
    ask(pool, [user('이 초록을 일본어로 번역해 줘')])
    assert 'block_cjk' not in server.bodies()[-1]
    ask(pool, [user('이 초록을 일본어로 번역해 줘'), ANSWER, user('조금 더 자연스럽게 다듬어 줘')])
    assert 'block_cjk' not in server.bodies()[-1]  # the follow-up still wants Japanese


def test_messages_become_openai_messages_with_call_ids(tmp_path):
    from app.mlx_runtime import convert_messages
    messages = [
        {'role': 'system', 'content': 'guide'},
        {'role': 'user', 'content': 'find <|image_pad|>papers'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': {'name': 'a', 'arguments': {'query': '당뇨'}}},
            {'type': 'function', 'function': {'name': 'b', 'arguments': {'pmids': ['1']}}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"pmids": []}'},
        {'role': 'tool', 'name': 'b', 'content': '{"articles": []}'},
        {'role': 'assistant', 'content': '<tool_call>{bad}</tool_call>'},
        {'role': 'tool', 'name': 'tool_error', 'content': '{"error": "invalid_tool_call"}'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'}]}]
    chat, templated, images = convert_messages(spec(), messages)
    assert images == 0 and chat[1] == {'role': 'user', 'content': 'find  papers'}
    assert chat[2]['tool_calls'] == [
        {'id': 'call_1', 'type': 'function', 'function': {'name': 'a', 'arguments': '{"query": "당뇨"}'}},
        {'id': 'call_2', 'type': 'function', 'function': {'name': 'b', 'arguments': '{"pmids": ["1"]}'}}]
    assert [m.get('tool_call_id') for m in chat if m['role'] == 'tool'] == ['call_1', 'call_2', None]
    assert chat[6]['name'] == 'tool_error' and chat[7]['content'] == 'first\n\nsecond'
    # The chat template iterates the arguments, so the counting copy keeps them as objects.
    assert templated[2]['tool_calls'][0]['function']['arguments'] == {'query': '당뇨'}


def test_no_message_of_any_role_carries_a_vision_control_token(tmp_path):
    """One would fail mlx-vlm's shared GPU loop, and with it every answer in progress."""
    from app.mlx_runtime import VISION, convert_messages
    messages = [
        {'role': 'system', 'content': 'guide <|vision_start|><|video_pad|>'},
        {'role': 'user', 'content': 'find <|vision_<|image_pad|>start|> papers'},  # one token inside another
        {'role': 'assistant', 'content': 'searching <|video_pad|>', 'tool_calls': [{'type': 'function', 'function': {
            'name': 'pubmed_search_articles', 'arguments': {
                'query': 'a<|vision_start|><|video_pad|>b', 'filters': {'terms': ['<|image_pad|>x'], 'max': 5}}}}]},
        {'role': 'tool', 'name': 'pubmed_search_articles', 'content': '{"title": "<|vision_end|>t"}'},
        {'role': 'assistant', 'content': 'done <|image_pad|>'}]
    chat, templated, images = convert_messages(spec(), messages)
    for converted in (chat, templated):
        assert not VISION.search(json.dumps(converted, ensure_ascii=False))
    assert [message.get('content') for message in chat] == [
        'guide   ', 'find <|vision_ start|> papers', 'searching  ', '{"title": " t"}', 'done  ']
    assert json.loads(chat[2]['tool_calls'][0]['function']['arguments']) == \
        templated[2]['tool_calls'][0]['function']['arguments'] == {'query': 'a  b', 'filters': {'terms': [' x'], 'max': 5}}
    pool, server, tokenizer = make_pool(tmp_path, Server(sse({'content': 'ok'})))
    schema = json.loads(json.dumps(SEARCH).replace('Search', 'Search <|vision_start|>PubMed'))  # from the MCP server
    ask(pool, messages, tools=[schema])
    assert not VISION.search(json.dumps(server.bodies()[-1], ensure_ascii=False))
    assert server.bodies()[-1]['tools'][0]['function']['description'] == tokenizer.calls[-1]['tools'][0][
        'function']['description'] == 'Search  PubMed'


def test_nested_vision_tokens_are_replaced_in_one_quick_pass():
    """Removing a token can join the text around it into the next one, so removal took one pass per nesting
    level: seconds for a 400 KB message, on a worker thread that Stop cannot end. A space joins into no token."""
    from app.images import VISION_TOKENS
    from app.mlx_runtime import VISION, strip_vision
    assert not any(' ' in token for token in VISION_TOKENS)
    for outer in VISION_TOKENS:  # every token inside every other, at every cut
        for inner in VISION_TOKENS:
            for cut in range(1, len(outer)):
                assert not VISION.search(strip_vision(outer[:cut] + inner + outer[cut:]))
    nested = '<|image' * 30000 + '<|image_pad|>' + '_pad|>' * 30000  # about 390 KB, 30,000 levels deep
    started = time.monotonic()
    cleaned = strip_vision([{'role': 'tool', 'content': nested}])
    assert time.monotonic() - started < 0.5
    assert cleaned[0]['content'] == '<|image' * 30000 + ' ' + '_pad|>' * 30000 and not VISION.search(
        cleaned[0]['content'])


def test_images_are_resized_like_on_the_transformers_path_and_counted(tmp_path):
    import base64
    import io
    from PIL import Image
    from app.images import MODEL_MAX_PIXELS
    from app.mlx_runtime import IMAGE_TOKENS, convert_messages
    from tests.test_images import data_url
    message = {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': data_url(size=(2000, 1000))}},
                                           {'type': 'text', 'text': '무엇이 보이나요?'}]}
    chat, templated, images = convert_messages(spec(), [message])
    url = chat[0]['content'][0]['image_url']['url']
    with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
        assert image.width * image.height <= MODEL_MAX_PIXELS and abs(image.width / image.height - 2) < 0.01
    assert templated[0]['content'] == [{'type': 'image'}, {'type': 'text', 'text': '무엇이 보이나요?'}]
    assert images == 1
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': 'ok'})), prompt=32768 - IMAGE_TOKENS - 99)
    ask(pool, [message], max_tokens=100)  # the template plus one image's tokens leave exactly 100
    assert server.bodies()[-1]['max_tokens'] == 100
    with pytest.raises(ServiceError) as e:
        convert_messages(spec('qwen3.5-9b', vision=False), [message])
    assert e.value.code == 'vision_unsupported'
    with pytest.raises(ServiceError) as e:
        convert_messages(spec(), [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.png'}}]}])
    assert e.value.code == 'invalid_image'


def test_chat_service_streams_an_mlx_answer_with_loading_status(tmp_path, monkeypatch):
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    from app.catalog import load_catalog
    from app.chat_service import ChatService
    from app.model_manager import ModelManager
    from app.schemas import ChatRequest
    from tests.test_chat_service import PubMed
    pool, server, _ = make_pool(tmp_path, Server(sse({'reasoning': '생각'}, {'content': '답입니다'})))
    manager = ModelManager(pool.settings, load_catalog(), pool=pool)
    request = ChatRequest(messages=[{'role': 'user', 'content': '질문'}], use_pubmed=False, stream=True,
                          reasoning_effort='light')
    events = []
    asyncio.run(ChatService(manager, PubMed()).stream(request, events.append, threading.Event(), 'user-3-1760000000'))
    assert events[0] == {'type': 'status', 'stage': 'loading_model', 'model': 'qwen3.5-9b'}
    assert {'type': 'status', 'stage': 'generating'} in events
    assert ''.join(e['content'] for e in events if e['type'] == 'reasoning') == '생각'
    assert ''.join(e['content'] for e in events if e['type'] == 'delta') == '답입니다'
    final = events[-1]['choices'][0]
    assert final['message'] == {'role': 'assistant', 'content': '답입니다', 'reasoning_content': '생각'}
    assert events[-1]['model'] == 'qwen3.5-9b'
    assert [r.headers.get('x-apc-tenant') for r in server.requests if r.url.path == '/v1/chat/completions'] == \
        [None, 'user-3-1760000000']
    events = []
    asyncio.run(ChatService(manager, PubMed()).stream(request, events.append, threading.Event()))
    assert not any(e.get('stage') in ('loading_model', 'queued') for e in events)  # ready, and it batches
    manager.close()
    assert pool.launcher.started[0][3].signals == [signal.SIGTERM]


def test_the_app_starts_its_sidecars_in_the_background_and_stops_them(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.catalog import load_catalog
    from app.chat_service import GUIDANCE
    from app.main import create_app
    from app.model_manager import ModelManager
    from tests.test_chat_service import PubMed
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    pool, server, _ = make_pool(tmp_path, Server(sse({'content': '안녕하세요'}), refuse_health=20))
    manager = ModelManager(pool.settings, load_catalog(), pool=pool)
    with TestClient(create_app(pool.settings, manager, PubMed())) as c:
        states = [c.get('/v1/models').json()['data'][0]['state']]  # the page answers while the model loads
        while states[-1] != 'ready' and len(states) < 500:
            time.sleep(0.01)
            states.append(c.get('/v1/models').json()['data'][0]['state'])
        assert states[0] in ('stopped', 'starting') and states[-1] == 'ready'
        assert c.get('/health').json()['ready_for_chat'] is True
        r = c.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': '안녕'}], 'use_pubmed': False})
        assert r.json()['model'] == 'qwen3.5-9b' and r.json()['choices'][0]['message']['content'] == '안녕하세요'
        deadline = time.monotonic() + 5
        while len(server.bodies(stream=False)) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        plain, pubmed = server.bodies(stream=False)
        assert pubmed['messages'][0] == {'role': 'system', 'content': GUIDANCE}  # local mode caches it after the start
    assert pool.launcher.started[0][3].signals == [signal.SIGTERM]
    assert state(pool) == 'stopped' and not (tmp_path / 'sockets').exists()


def test_a_model_is_not_stopped_as_idle_while_its_prompt_is_prepared(tmp_path, monkeypatch):
    """ChatService decides whether to show "loading" from the model's state just before generate(); the idle check
    must not stop the model in between, or it restarts without that notice."""
    monkeypatch.setattr('app.settings.mlx_available', lambda: True)
    import app.mlx_runtime as runtime
    from app.catalog import load_catalog
    from app.chat_service import ChatService
    from app.model_manager import ModelManager
    from app.schemas import ChatRequest
    from tests.test_chat_service import PubMed
    pool, _, _ = make_pool(tmp_path, Server(sse({'content': '네'})), models=NINE_FOUR)
    pool.idle_check_seconds = 0.01
    manager = ModelManager(pool.settings, load_catalog(), pool=pool)
    judge = runtime.blocks_cjk

    def slow_judge(messages):  # generate()'s first step, slow as with a large prompt
        time.sleep(0.2)
        return judge(messages)
    monkeypatch.setattr(runtime, 'blocks_cjk', slow_judge)
    events = []

    async def main():
        async with anyio.create_task_group() as app:
            await app.start(pool.run, None, ['qwen3.5-9b'])
            await pool.ensure('qwen3.5-4b')
            pool.sidecars['qwen3.5-4b'].used -= IDLE - 0.05  # crosses MLX_IDLE_MINUTES 50 ms from now
            request = ChatRequest(model='qwen3.5-4b', messages=[{'role': 'user', 'content': '안녕'}], use_pubmed=False,
                                  stream=True)
            await ChatService(manager, PubMed()).stream(request, events.append, threading.Event())
            app.cancel_scope.cancel()
    asyncio.run(main())
    pool.close()
    assert len(launched(pool, 'qwen3.5-4b')) == 1  # kept, not stopped and started again
    assert events[-1]['choices'][0]['message']['content'] == '네'
