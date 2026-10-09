"""MLX models served by mlx_vlm.server sidecars: one process per model, each on its own private Unix socket.

A sidecar keeps its model in memory, batches requests itself and caches prompt prefixes (APC). This module
starts, watches and stops the processes (a model that was not preloaded also after MLX_IDLE_MINUTES without a
request) and turns ModelManager.generate calls into streaming /v1/chat/completions requests with the same
contract as the Transformers path, so app/chat_service.py works unchanged.
"""
import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import anyio
import httpx
from app.catalog import ROOT, ModelSpec, files_ready
from app.images import MODEL_MAX_PIXELS, VISION_TOKENS, decode_image
from app.model_manager import THINKING_BUDGETS, GenerationResult, plan_output
from app.settings import ServiceError, Settings

log = logging.getLogger('app.mlx')
READY_TIMEOUT = 180  # seconds for a sidecar to load its model and answer /health
POLL_SECONDS = 0.5
STOP_TIMEOUT = 10
PREFIX_TIMEOUT = 10  # seconds to fetch the PubMed prompt that is cached after a start
CANCEL_POLL_SECONDS = 0.1  # how soon Stop ends an answer whose prompt is still being read
IDLE_CHECK_SECONDS = 30  # how often models are checked against MLX_IDLE_MINUTES
LOG_LIMIT = 20 * 1024 * 1024  # a log larger than this is emptied when its sidecar starts
SOCKET_PATH_LIMIT = 104  # bytes in an AF_UNIX address on macOS, the smallest of the supported systems
# A long prompt is read for a while before the first token arrives; after that tokens stream steadily.
STREAM_TIMEOUT = httpx.Timeout(600, connect=5)
# Images are resized to MODEL_MAX_PIXELS as on the Transformers path, and Qwen3.5 merges 32x32 pixels into one
# token, so an image costs at most about this many prompt tokens.
IMAGE_TOKENS = 280
# The server cannot extend an assistant message, so the rest is asked for. The model repeats the answer's last
# words first and the app drops them, so a word cut in half is finished and the space between words is right
# ("…것이" + " 가장 …", "…것이 고" + "려해야 …") instead of a new sentence glued on ("…것이가장 …").
CONTINUE_PROMPT = ('Continue your previous answer exactly where it stopped. Begin your reply by repeating its last '
                   'words «{tail}» exactly as written, then go on from there. Do not add any preface and do not '
                   'repeat anything earlier.')
CONTINUE_TAIL = 30  # characters of the answer quoted, from the start of a word
HELLO = [{'role': 'user', 'content': '안녕하세요'}]  # the warm-up question
HANGUL = re.compile('[가-힣ᄀ-ᇿ㄰-㆏]')
CJK = re.compile('[぀-ヿ㐀-䶿一-鿿豈-﫿]')  # kana and Chinese characters
# A request for Chinese or Japanese writing names the script as what to write: 일본어로 번역, 간체로 써 줘, 한자 표기·뜻,
# "into Japanese", "translate … Chinese", kanji/hanja. A mere mention ("중국어 논문은 빼고", "한자나 중국어를 섞지
# 마세요", "Do not use Chinese characters") asks for nothing, and 한자리·한자세·한문장·시간체크 are other words.
SCRIPTS = '일본어|중국어|한자|한문|간체|번체|히라가나|가타카나'
ASKS_KO = re.compile(f'(?:{SCRIPTS})(?:\\s*(?:으로|로)|\\s*(?:의\\s*)?(?:번역|표기|병기|발음|읽는\\s*법|읽기|뜻|단어|글자|표현|'
                     '인사|이름|원문|버전))')
NOT_KO = re.compile('\\s*\\S{0,6}?(?:지\\s*(?:마|말|않)|금지)')  # its own verb negated: 한자로 쓰지 마세요
ASKS_EN = re.compile(r'\b(?:in|into|to)\s+(?:simplified\s+|traditional\s+)?(?:japanese|chinese|mandarin|cantonese)\b'
                     r'|\btranslat\w*\b[^.!?\n]{0,60}?\b(?:japanese|chinese|mandarin|cantonese)\b'
                     r'|\b(?:japanese|chinese|mandarin|cantonese)\s*:|\b(?:kanji|hanja|hanzi|hiragana|katakana|pinyin)\b',
                     re.I)
# "Do not answer in Chinese"; a "but" turns it around: "Not Chinese but Japanese: …"
NOT_EN = re.compile(r"\b(?:not|never|don['’]?t|avoid|no)\b(?:(?!\b(?:but|instead|rather)\b)[^.!?\n])*\Z", re.I)
ATTACHED = re.compile('\\[첨부 파일: [^\\]\\n]*\\]\\n```\\n.*?\\n```', re.S)  # files the web UI joins to a question
SCAN = 4000  # request words sit at a question's start or end; pasted text is not read whole
VISION = re.compile('|'.join(map(re.escape, VISION_TOKENS)))
TOOL_CALL = re.compile(r'<tool_call>.*?</tool_call>', re.S)
TOO_LONG = re.compile(r'context tokens|MAX_KV_SIZE|too long|context window', re.I)
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


class SidecarFailed(Exception):
    """A sidecar did not come up; the message becomes its state's error text."""


class Unreachable(Exception):
    """No connection to a sidecar at all, so the request can be retried after a restart."""


class GroupProcess:
    """A sidecar in its own session: signals reach it and anything it starts, but a Ctrl+C in the app's
    terminal does not, so the app decides when it stops."""

    def __init__(self, popen: subprocess.Popen):
        self.popen, self.pid = popen, popen.pid

    def poll(self):
        return self.popen.poll()

    def wait(self, timeout=None):
        return self.popen.wait(timeout)

    def signal(self, sig) -> None:
        if self.popen.poll() is None:
            try:
                os.killpg(self.popen.pid, sig)
            except ProcessLookupError:
                pass


def launch(command: list[str], env: dict[str, str], log_path: Path) -> GroupProcess:
    with open(log_path, 'ab') as log_file:
        return GroupProcess(subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log_file,
                                             stderr=subprocess.STDOUT, start_new_session=True))


def socket_dir(longest: str) -> Path:
    """A new directory only this user can open (0700), for one app run's sidecar sockets.

    uvicorn lets everyone read and write a socket it creates, so the directory is what keeps other local users
    away from the models and the key. A long TMPDIR would make a socket path longer than AF_UNIX allows; /tmp
    is short.
    """
    path = Path(tempfile.mkdtemp(prefix='pubmed-mlx-'))
    if len(os.fsencode(path / longest)) < SOCKET_PATH_LIMIT:
        return path
    path.rmdir()
    return Path(tempfile.mkdtemp(prefix='pubmed-mlx-', dir='/tmp'))


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def last_log_line(path: Path) -> str:
    try:
        with open(path, 'rb') as file:
            file.seek(max(0, file.seek(0, os.SEEK_END) - 4096))
            lines = ANSI.sub('', file.read().decode('utf-8', 'replace')).splitlines()
    except OSError:
        return ''
    return next((line.strip()[:200] for line in reversed(lines) if line.strip()), '')


@dataclass(eq=False)
class Start:
    """One start of a sidecar, shared by every request waiting for its model; error is set when it failed."""
    done: anyio.Event = field(default_factory=anyio.Event)
    error: ServiceError | None = None


@dataclass(eq=False)
class Sidecar:
    spec: ModelSpec
    index: int  # it serves on <the run's socket directory>/<index>.sock
    path: Path
    state: str = 'stopped'  # missing (files not prepared), stopped, starting, ready or failed
    error: str | None = None
    process: object = None
    key: str = ''  # random per start; the server refuses requests without it
    socket: Path | None = None  # set by its first start
    start: Start | None = None  # the start under way while MLXPool.run owns the starts
    # One start or idle stop at a time, so a new process never shares the socket with one still stopping.
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    in_flight: int = 0  # requests using it now
    used: float = 0.0  # time.monotonic() when a request last started or ended, or when it became ready

    @property
    def headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.key}'}

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @contextmanager
    def in_use(self):
        """A request needs the model: it is not stopped as idle until the request is over."""
        self.in_flight += 1
        self.used = time.monotonic()
        try:
            yield
        finally:
            self.in_flight -= 1
            self.used = time.monotonic()


def strip_vision(value):
    """The value with every vision control token in its text replaced by a space, however deeply nested.

    One in any message (system, answer, tool result or call argument) would be read as an image placeholder,
    which fails mlx-vlm's shared GPU loop and with it every answer in progress. One pass is enough and stays
    linear: no token holds a space, so the text around a replaced token never joins into a new one (removing
    tokens instead could, one nesting level per pass).
    """
    if isinstance(value, str):
        return VISION.sub(' ', value)
    if isinstance(value, dict):
        return {strip_vision(key): strip_vision(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_vision(item) for item in value]
    return value


def own_error(error: ServiceError) -> ServiceError:
    """A copy of a shared start's error for one waiting request; ChatService adds that request's tool calls."""
    return ServiceError(error.code, error.message, error.status_code, dict(error.details), dict(error.headers))


async def watch_cancel(cancel: threading.Event, scope: anyio.CancelScope) -> None:
    while not cancel.is_set():
        await anyio.sleep(CANCEL_POLL_SECONDS)
    scope.cancel()


def model_image(url: str) -> str:
    """The image as a PNG data URL no larger than on the Transformers path (about 256 tokens on Qwen3.5).

    The server's own processor takes up to 16.7 megapixels, about 16k tokens per image.
    """
    image = decode_image(url)
    scale = (MODEL_MAX_PIXELS / (image.width * image.height)) ** 0.5
    if scale < 1:
        from PIL import Image
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                             Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def convert_messages(spec: ModelSpec, messages: list[dict]) -> tuple[list[dict], list[dict], int]:
    """The conversation as OpenAI messages for the server, as chat-template input for counting its tokens,
    and the number of images.

    Tool calls get ids call_<n> and JSON-text arguments, and the tool results after them the matching ids
    in order; correction notes (name tool_error) answer no call. No text keeps a vision control token.
    """
    if not spec.vision and any(isinstance(message.get('content'), list) and any(
            part['type'] == 'image_url' for part in message['content']) for message in messages):
        raise ServiceError('vision_unsupported', f'{spec.id} cannot read images', 422)
    chat, templated, pending, images, calls = [], [], [], 0, 0
    for message in messages:
        role, content = message['role'], message.get('content')
        if role == 'assistant' and message.get('tool_calls'):
            sent, kept = [], []
            for call in message['tool_calls']:
                function = call.get('function', call)
                name, arguments = strip_vision(function['name']), strip_vision(function['arguments'])
                calls += 1
                sent.append({'id': f'call_{calls}', 'type': 'function', 'function': {
                    'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
                kept.append({'type': 'function', 'function': {'name': name, 'arguments': arguments}})
            pending = [item['id'] for item in sent]
            text = strip_vision(content or '')
            chat.append({'role': role, 'content': text, 'tool_calls': sent})
            templated.append({'role': role, 'content': text, 'tool_calls': kept})
        elif role == 'tool':
            item = {'role': role, 'name': strip_vision(message.get('name')), 'content': strip_vision(content)}
            templated.append(dict(item))
            if pending:
                item['tool_call_id'] = pending.pop(0)
            chat.append(item)
        elif role == 'user' and isinstance(content, list):
            sent, kept = [], []
            for part in content:
                if part['type'] == 'image_url':
                    images += 1
                    sent.append({'type': 'image_url', 'image_url': {'url': model_image(part['image_url']['url'])}})
                    kept.append({'type': 'image'})
                else:
                    text = {'type': 'text', 'text': strip_vision(part['text'])}
                    sent.append(text)
                    kept.append(text)
            if all(part['type'] == 'text' for part in sent):  # joined as on the Transformers path
                sent = kept = '\n\n'.join(part['text'] for part in sent)
            chat.append({'role': role, 'content': sent})
            templated.append({'role': role, 'content': kept})
        else:
            text = strip_vision(content)
            chat.append({'role': role, 'content': text})
            templated.append({'role': role, 'content': text})
    return chat, templated, images


def message_text(message: dict) -> str:
    content = message.get('content')
    if isinstance(content, list):
        return ' '.join(part.get('text', '') for part in content if part.get('type') == 'text')
    return content if isinstance(content, str) else ''


def answer_tail(text: str) -> str:
    """The last words of an answer to quote, starting at a word when the answer is longer."""
    tail = text[-CONTINUE_TAIL:]
    if len(text) > CONTINUE_TAIL and not text[-CONTINUE_TAIL - 1].isspace() and re.search(r'\s', tail):
        tail = tail[re.search(r'\s', tail).end():]
    return tail


def repeated_prefix(tail: str, text: str) -> int:
    """How many characters at the start of text repeat the end of tail (the longest ending of tail that starts
    at a word, so a stray character does not count); 0 when the model did not repeat it."""
    stripped = text.lstrip()
    for start in range(len(tail)):
        piece = tail[start:].strip()
        if len(piece) >= 2 and (start == 0 or tail[start - 1].isspace()) and stripped.startswith(piece):
            return len(text) - len(stripped) + len(piece)
    return 0


def ends(text: str, size: int) -> str:
    """The start and end of a long text, which is all these checks need to read."""
    return text if len(text) <= 2 * size else text[:size] + '\n' + text[-size:]


def asks_cjk(text: str) -> bool:
    """Whether the text asks for Chinese or Japanese writing; attached files are not read for it (an English
    abstract about "Chinese adults" asks for nothing)."""
    text = ends(ATTACHED.sub(' ', text), SCAN)
    return (any(not NOT_KO.match(text, found.end()) for found in ASKS_KO.finditer(text))
            or any(not NOT_EN.search(text, max(0, found.start() - 40), found.start())
                   for found in ASKS_EN.finditer(text)))


def blocks_cjk(messages: list[dict]) -> bool:
    """Korean questions get no stray Chinese or Japanese characters, unless a user message holds such characters
    or any user or system message asks for such writing (a translation into Japanese, a word's Hanja), so that
    a follow-up such as "더 자연스럽게 다듬어 줘" keeps what an earlier question asked for. Runs on a worker
    thread and reads at most the ends of long texts, so a huge message cannot stall the server."""
    asked = [message_text(message) for message in messages if message['role'] == 'user']
    systems = [message_text(message) for message in messages if message['role'] == 'system']
    return (bool(asked) and bool(HANGUL.search(ends(asked[-1], SCAN)))
            and not any(CJK.search(ends(text, 5 * SCAN)) for text in asked)
            and not any(asks_cjk(text) for text in [*asked, *systems]))


def raw_text(reasoning: str, content: str, thinking: bool, done: bool = True) -> str:
    """Server output in the Transformers path's form; '</think>' appears once the answer has started."""
    if not thinking:
        return content
    return '<think>' + reasoning + ('</think>' + content if content or done else '')


def move_tool_calls(reasoning: str, content: str) -> tuple[str, str]:
    """Qwen3.5 sometimes writes its tool call inside the reasoning; complete calls there belong to the answer."""
    blocks = TOOL_CALL.findall(reasoning)
    if not blocks or '<tool_call>' in content:
        return reasoning, content
    return TOOL_CALL.sub('', reasoning).strip(), (content.rstrip() + '\n' + '\n'.join(blocks)).lstrip()


def server_call(call: dict) -> str:
    """A call the server parsed itself (tool_parsing server), as markup app/chat_service.py reads."""
    function = call.get('function') or {}
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass  # stays text, which validation reports back to the model
    return '<tool_call>' + json.dumps({'name': function.get('name'), 'arguments': arguments},
                                      ensure_ascii=False) + '</tool_call>'


def http_error(status: int, text: str) -> ServiceError:
    try:
        detail = json.loads(text).get('detail') or text
    except (ValueError, AttributeError):
        detail = text
    detail = str(detail)[:300]
    if status == 400 and TOO_LONG.search(detail):
        return ServiceError('context_overflow', 'Input plus requested output exceeds context budget', 413,
                            {'detail': detail})
    return ServiceError('generation_failed', f'The model server answered HTTP {status}: {detail}', 502)


class MLXPool:
    """One mlx_vlm.server process per enabled MLX model; tests replace the launcher, the HTTP transport and the
    socket directory."""
    poll_seconds, ready_timeout, stop_timeout = POLL_SECONDS, READY_TIMEOUT, STOP_TIMEOUT
    prefix_timeout, idle_check_seconds = PREFIX_TIMEOUT, IDLE_CHECK_SECONDS

    def __init__(self, settings: Settings, specs: list[ModelSpec], *, launcher=launch,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.launcher, self.transport = launcher, transport
        self.load_tokenizer, self.socket_dir = load_tokenizer, socket_dir
        self.sidecars = {spec.id: Sidecar(spec, n, settings.models_dir / spec.directory)
                         for n, spec in enumerate(specs)}
        for sidecar in self.sidecars.values():
            if not files_ready(sidecar.path):
                sidecar.state = 'missing'
        self.prefix = None  # set by run(): the PubMed prompt start to cache after each start
        self.preloaded: frozenset[str] = frozenset()  # set by run(): started with the app, never stopped as idle
        self._group = None  # run()'s task group, which owns every start while the app runs
        self._sockets = None  # this run's socket directory, made by the first start
        self._tokenizers: dict[str, object] = {}
        self._tokenizer_lock = threading.Lock()  # fast tokenizers must not be used by two threads at once
        self._closed = False

    def __contains__(self, model_id: str) -> bool:
        return model_id in self.sidecars

    def _client(self, sidecar: Sidecar, timeout: httpx.Timeout = STREAM_TIMEOUT) -> httpx.AsyncClient:
        # The sidecar's own socket; trust_env=False: no proxy setting ever applies to these requests.
        transport = self.transport or httpx.AsyncHTTPTransport(uds=str(sidecar.socket))
        return httpx.AsyncClient(transport=transport, base_url='http://mlx', trust_env=False, timeout=timeout)

    def status(self) -> dict[str, dict]:
        return {model_id: {'state': self._state(sidecar), 'runtime': 'mlx', 'label': sidecar.spec.label,
                           'error': sidecar.error} for model_id, sidecar in self.sidecars.items()}

    def _state(self, sidecar: Sidecar) -> str:
        if sidecar.state == 'ready' and not sidecar.alive():
            sidecar.state, sidecar.error = 'failed', self._exit_reason(sidecar)
            log.warning('%s stopped: %s', sidecar.spec.id, sidecar.error)
        elif sidecar.state == 'missing' and files_ready(sidecar.path):
            sidecar.state = 'stopped'
        return sidecar.state

    async def run(self, prefix=None, preload_ids=(), *, task_status=anyio.TASK_STATUS_IGNORED) -> None:
        """Own every start until cancelled at shutdown, so a request that leaves while its model starts only
        stops waiting. The models in preload_ids start now and stay loaded; the others start on their first
        question and are stopped after MLX_IDLE_MINUTES without one. prefix gives the PubMed prompt start to
        cache after each start (local mode)."""
        self.prefix, self.preloaded = prefix, frozenset(preload_ids)
        try:
            async with anyio.create_task_group() as group:
                self._group = group
                group.start_soon(self.start_all, self.preloaded)
                if self.settings.mlx_idle_minutes:
                    group.start_soon(self._stop_idle)
                task_status.started()
                await anyio.sleep_forever()
        finally:
            self._group, self._closed = None, True  # the app is stopping: nothing starts again

    async def start_all(self, ids=None) -> None:
        """Start every model (or those in ids) one at a time, default first; a failure stays in its state."""
        for model_id in self.sidecars:
            if ids is not None and model_id not in ids:
                continue
            try:
                await self.ensure(model_id)
            except ServiceError:
                pass  # a request for the model tries again
            except Exception:  # a background task: log it rather than end the app's lifespan with it
                log.exception('%s could not start', model_id)

    async def ensure(self, model_id: str) -> Sidecar:
        """The model's sidecar, ready; started now when it is stopped, has crashed or failed before."""
        sidecar = self.sidecars[model_id]
        if self._state(sidecar) == 'ready':
            return sidecar
        if self._group is None:  # no app around the pool (tests, scripts): the caller starts it
            async with sidecar.lock:  # one start serves every request waiting for it
                if self._state(sidecar) != 'ready':
                    await self._start(sidecar)
            return sidecar
        start = sidecar.start
        if start is None:
            start = sidecar.start = Start()
            self._group.start_soon(self._own_start, sidecar, start)
        await start.done.wait()  # a request that leaves stops waiting here; the start goes on
        if start.error is not None:
            raise own_error(start.error)
        return sidecar

    async def _own_start(self, sidecar: Sidecar, start: Start) -> None:
        try:
            async with sidecar.lock:  # an idle stop under way ends first
                await self._start(sidecar)
        except ServiceError as exc:
            start.error = exc
        except Exception as exc:  # a task in the app's group must never end the app
            log.exception('%s could not start', sidecar.spec.id)
            start.error = ServiceError('model_unavailable', f'{sidecar.spec.label} could not be started: {exc}', 503)
        finally:
            if start.error is None and sidecar.state != 'ready':  # cancelled: the app is stopping
                start.error = ServiceError('model_unavailable', 'The server is shutting down', 503)
            sidecar.start = None
            start.done.set()

    async def _start(self, sidecar: Sidecar) -> None:
        if self._closed:
            raise ServiceError('model_unavailable', 'The server is shutting down', 503)
        if not files_ready(sidecar.path):
            sidecar.state, sidecar.error = 'missing', None
            log.warning('%s is not prepared; run: uv run python scripts/prepare_models.py', sidecar.spec.id)
            raise ServiceError('model_missing', 'Run: uv run python scripts/prepare_models.py', 503)
        if sidecar.process is not None:
            await anyio.to_thread.run_sync(self._stop, sidecar)  # what is left of a crashed or stuck one
        sidecar.state, sidecar.error = 'starting', None
        started = time.monotonic()
        try:
            sidecar.socket = self._socket(sidecar)
            sidecar.key = secrets.token_urlsafe(32)  # through the environment: argv is visible to every user
            log_path = await anyio.to_thread.run_sync(self._prepare_files, sidecar)
            env = self._env(sidecar)
            with anyio.CancelScope(shield=True):  # a launched process is always recorded, so it can be stopped
                sidecar.process = await anyio.to_thread.run_sync(self.launcher, self._command(sidecar), env,
                                                                 log_path)
            log.info('starting %s on %s, log in %s', sidecar.spec.id, sidecar.socket, log_path)
            await self._wait_ready(sidecar)
            await self._warm_up(sidecar)
        except (SidecarFailed, OSError, ValueError) as exc:
            sidecar.state, sidecar.error = 'failed', str(exc)
            log.warning('%s could not start: %s', sidecar.spec.id, exc)
            if sidecar.process is not None:
                sidecar.process.signal(signal.SIGTERM)
            raise ServiceError('model_unavailable', f'{sidecar.spec.label} could not be started: {exc}',
                               503) from exc
        except BaseException:  # cancelled: the app is stopping, or (without run()) the request that started it left
            sidecar.state = 'stopped'
            if sidecar.process is not None:
                sidecar.process.signal(signal.SIGTERM)
            raise
        sidecar.state, sidecar.used = 'ready', time.monotonic()  # idle from now, not since it last stopped
        log.info('%s ready in %.1f s', sidecar.spec.id, time.monotonic() - started)
        if self.prefix is not None and self._group is not None:
            self._group.start_soon(self._warm_prefix, sidecar)

    async def _stop_idle(self) -> None:
        """Stop the models that were not preloaded once they have had no request for MLX_IDLE_MINUTES."""
        limit = self.settings.mlx_idle_minutes * 60
        while True:
            await anyio.sleep(self.idle_check_seconds)
            for sidecar in self.sidecars.values():
                if self._idle(sidecar, limit):
                    try:
                        await self._retire(sidecar, limit)
                    except Exception:  # a task in the app's group must never end the app
                        log.exception('%s could not be stopped', sidecar.spec.id)

    def _idle(self, sidecar: Sidecar, limit: float) -> bool:
        return (sidecar.spec.id not in self.preloaded and not sidecar.in_flight and sidecar.start is None
                and self._state(sidecar) == 'ready' and time.monotonic() - sidecar.used > limit)

    async def _retire(self, sidecar: Sidecar, limit: float) -> None:
        async with sidecar.lock:  # a start asked for meanwhile waits until this process is gone
            if not self._idle(sidecar, limit):  # a request came while the lock was taken
                return
            minutes = (time.monotonic() - sidecar.used) // 60
            sidecar.state = 'stopped'  # from here on a request starts it again
            await anyio.to_thread.run_sync(self._stop, sidecar)
        log.info('stopped %s after %d min idle', sidecar.spec.id, minutes)

    def _socket(self, sidecar: Sidecar) -> Path:
        if self._sockets is None:  # one directory per app run, made by the first start and removed by close()
            self._sockets = self.socket_dir(f'{len(self.sidecars) - 1}.sock')
        return self._sockets / f'{sidecar.index}.sock'

    async def _wait_ready(self, sidecar: Sidecar) -> None:
        """Until /health names the model; the server creates its socket only once the model is loaded."""
        deadline = time.monotonic() + self.ready_timeout
        async with self._client(sidecar, httpx.Timeout(5)) as client:
            while True:
                if not sidecar.alive():
                    raise SidecarFailed(self._exit_reason(sidecar))
                if self._closed:
                    raise SidecarFailed('the app is shutting down')
                try:
                    response = await client.get('/health', headers=sidecar.headers)
                    if response.status_code == 200 and response.json().get('loaded_model') == str(sidecar.path):
                        return
                except (httpx.HTTPError, ValueError, AttributeError):
                    pass
                if time.monotonic() > deadline:
                    raise SidecarFailed(f'not ready after {self.ready_timeout} s')
                await anyio.sleep(self.poll_seconds)

    async def _warm_up(self, sidecar: Sidecar) -> None:
        """A one-token answer compiles the kernels and builds the CJK mask, and the tokenizer that counts prompt
        tokens is loaded, so the first question waits for neither."""
        await self._complete(sidecar, {'messages': HELLO})
        try:
            await anyio.to_thread.run_sync(self._tokenizer, sidecar.spec)
        except Exception as exc:
            log.warning('%s: tokenizer for the context budget not loaded (%s)', sidecar.spec.id, exc)

    async def _warm_prefix(self, sidecar: Sidecar) -> None:
        """Cache the system prompt and tool schemas every PubMed question starts with (local mode), after the
        model is ready: a question without PubMed never waits for PubMed."""
        try:
            with anyio.fail_after(self.prefix_timeout):  # PubMed may be slow or offline
                system, tools = await self.prefix()
            parser = {'tool_parser': 'longcat'} if sidecar.spec.tool_parsing == 'app' else {}
            await self._complete(sidecar, {'messages': [{'role': 'system', 'content': strip_vision(system)}, *HELLO],
                                           'tools': strip_vision(tools), **parser})  # as generate() sends them
            log.info('%s: PubMed prompt cached', sidecar.spec.id)
        except Exception as exc:  # the first PubMed question then reads it in full
            log.warning('%s: PubMed prompt not cached (%r)', sidecar.spec.id, exc)

    async def _complete(self, sidecar: Sidecar, extra: dict) -> None:
        body = {'model': str(sidecar.path), 'max_tokens': 1, 'stream': False, 'enable_thinking': False,
                'block_cjk': True, 'context_management': [], **extra}
        try:
            async with self._client(sidecar) as client:
                response = await client.post('/v1/chat/completions', json=body, headers=sidecar.headers)
        except httpx.HTTPError as exc:
            raise SidecarFailed(f'warm-up failed: {exc}') from exc
        if response.status_code != 200:
            raise SidecarFailed(f'warm-up answered HTTP {response.status_code}: {response.text[:200]}')

    def _prepare_files(self, sidecar: Sidecar) -> Path:
        """Create DATA_DIR/logs and DATA_DIR/kv-cache, empty an oversized log, and return the log's path."""
        data = self.settings.data_dir
        data.mkdir(mode=0o700, parents=True, exist_ok=True)
        (data / 'kv-cache').mkdir(mode=0o700, exist_ok=True)  # derived from chats: as private as chat.db
        (data / 'logs').mkdir(exist_ok=True)
        path = self._log_path(sidecar)
        if path.is_file() and path.stat().st_size > LOG_LIMIT:
            path.write_bytes(b'')
        return path

    def _log_path(self, sidecar: Sidecar) -> Path:
        return self.settings.data_dir / 'logs' / f'mlx-{sidecar.spec.id}.log'

    def _command(self, sidecar: Sidecar) -> list[str]:
        # The wrapper serves on the socket in MLX_SIDECAR_SOCKET only; the host guards in case that ever failed.
        return [sys.executable, '-m', 'app.mlx_sidecar', '--host', '127.0.0.1', '--model', str(sidecar.path),
                '--log-progress-interval', '0']

    def _env(self, sidecar: Sidecar) -> dict[str, str]:
        settings, cache = self.settings, self.settings.data_dir / 'kv-cache'
        # The MLX_SIDECAR_ variables are the app's to set, whatever its own environment holds.
        env = {name: value for name, value in os.environ.items() if not name.startswith('MLX_SIDECAR_')}
        env.update({'APC_ENABLED': '1', 'APC_CHECKPOINT_INTERVAL_TOKENS': str(settings.mlx_checkpoint_interval),
                    'APC_CHECKPOINT_ENTRIES': str(settings.mlx_checkpoint_entries),
                    'MLX_VLM_CACHE_HOME': str(cache),  # nothing derived from chats lands in ~/.cache/mlx-vlm
                    'MLX_VLM_SERVER_API_KEY': sidecar.key, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                    'PYTHONUNBUFFERED': '1', 'MLX_SIDECAR_PARENT_PID': str(os.getpid()),
                    'MLX_SIDECAR_SOCKET': str(sidecar.socket)})
        if settings.auth_enabled:  # image features are cached by the image's content, not by account
            env['MLX_SIDECAR_PRIVATE'] = '1'
        if settings.kv_cache_gb > 0:  # KV_CACHE_GB is shared by the models
            env.update(APC_DISK_ENABLED='1', APC_DISK_PATH=str(cache / sidecar.spec.id),
                       APC_DISK_MAX_GB=f'{settings.kv_cache_gb / len(self.sidecars):g}')
        else:
            env['APC_DISK_ENABLED'] = '0'
        return env

    def _exit_reason(self, sidecar: Sidecar) -> str:
        code = sidecar.process.poll() if sidecar.process is not None else None
        line = last_log_line(self._log_path(sidecar))
        return f'exited with code {code}' + (f': {line}' if line else '')

    def _stop(self, sidecar: Sidecar) -> None:
        process, sidecar.process = sidecar.process, None
        if process is None or process.poll() is not None:
            return
        process.signal(signal.SIGTERM)
        try:
            process.wait(self.stop_timeout)
        except subprocess.TimeoutExpired:
            process.signal(signal.SIGKILL)

    def close(self) -> None:
        """Stop every sidecar: SIGTERM to its process group, then SIGKILL after STOP_TIMEOUT seconds."""
        self._closed = True
        running = [sidecar.process for sidecar in self.sidecars.values() if sidecar.alive()]
        for process in running:
            process.signal(signal.SIGTERM)
        deadline = time.monotonic() + self.stop_timeout
        for process in running:
            try:
                process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.signal(signal.SIGKILL)
        for sidecar in self.sidecars.values():
            sidecar.process = None
            if sidecar.state != 'missing':
                sidecar.state = 'stopped'
        if self._sockets is not None:
            shutil.rmtree(self._sockets, ignore_errors=True)
            self._sockets = None

    def _tokenizer(self, spec: ModelSpec):
        with self._tokenizer_lock:
            tokenizer = self._tokenizers.get(spec.id)
            if tokenizer is None:
                path = self.settings.models_dir / spec.directory
                if not files_ready(path):
                    raise ServiceError('model_missing', 'Run: uv run python scripts/prepare_models.py', 503)
                tokenizer = self._tokenizers[spec.id] = self.load_tokenizer(path)
            return tokenizer

    def count_prompt(self, spec: ModelSpec, messages: list[dict], tools: list[dict], thinking: bool) -> int:
        """Prompt tokens by the model's own tokenizer and chat template, as the server will read them."""
        tokenizer = self._tokenizer(spec)
        with self._tokenizer_lock:
            text = tokenizer.apply_chat_template(messages, tools=tools or None, enable_thinking=thinking,
                                                 add_generation_prompt=True, tokenize=False)
            return len(tokenizer(text, add_special_tokens=False)['input_ids'])

    async def generate(self, spec: ModelSpec, messages: list[dict], tools: list[dict], max_tokens: int,
                       temperature: float | None = None, *, on_text=None, cancel: threading.Event | None = None,
                       effort: str = 'none', continue_final: bool = False, strict_budget: bool = False,
                       tenant: str | None = None) -> GenerationResult:
        """ModelManager.generate for a model in this pool."""
        if cancel is not None and cancel.is_set():
            return GenerationResult('', 'cancelled', 0, 0)
        # In use before the first await: ChatService has just decided whether to show "loading" from the model's
        # state, so the idle check must not stop it while the prompt is prepared. A stop decided just before ends
        # first, then ensure() starts the model again.
        with self.sidecars[spec.id].in_use():
            return await self._generate(spec, messages, tools, max_tokens, temperature, on_text=on_text, cancel=cancel,
                                        effort=effort, continue_final=continue_final, strict_budget=strict_budget,
                                        tenant=tenant)

    async def _generate(self, spec: ModelSpec, messages: list[dict], tools: list[dict], max_tokens: int,
                        temperature: float | None, *, on_text, cancel, effort: str, continue_final: bool,
                        strict_budget: bool, tenant: str | None) -> GenerationResult:
        block = await anyio.to_thread.run_sync(blocks_cjk, messages)
        repeat = None
        if continue_final:  # the server cannot extend an assistant message, so it is asked for the rest
            repeat = answer_tail(message_text(messages[-1]))
            messages = [*messages, {'role': 'user', 'content': CONTINUE_PROMPT.format(tail=repeat)}]
            tools, effort = [], 'none'
        tools = strip_vision(tools)  # the PubMed server's schemas are rendered into the prompt as well
        asked = effort != 'none' and spec.reasoning
        chat, templated, images = await anyio.to_thread.run_sync(convert_messages, spec, messages)
        try:
            prompt = await anyio.to_thread.run_sync(self.count_prompt, spec, templated, tools, asked)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError('generation_failed', f'Could not count the prompt tokens: {exc}', 502) from exc
        prompt += images * (IMAGE_TOKENS - 1)  # the template already counted one placeholder per image
        answer, budget = plan_output(spec.context_tokens or self.settings.max_context_tokens, prompt, max_tokens,
                                     THINKING_BUDGETS[effort] if asked else 0, strict_budget)
        thinking = asked and budget > 0  # no room left to reason: the answer comes after an empty reasoning
        sidecar = self.sidecars[spec.id]
        body = {'model': str(sidecar.path), 'messages': chat, 'stream': True,
                'stream_options': {'include_usage': True}, 'max_tokens': answer + budget, 'enable_thinking': thinking,
                **(spec.thinking_sampling if thinking else spec.sampling),
                # The server's default seed is fixed, so every regenerated answer would come out the same.
                'seed': secrets.randbelow(2 ** 31),
                # Never summarised by the server, which also skips a second pass over the prompt.
                'context_management': []}
        if temperature is not None:
            body['temperature'] = temperature
        if thinking:
            body['thinking_budget'] = budget
        if tools:
            body['tools'] = tools
            if spec.tool_parsing == 'app':
                # Markers Qwen never writes: the raw <tool_call> markup stays in the content for
                # app/chat_service.py, which repairs and corrects calls the server's parser would drop.
                body['tool_parser'] = 'longcat'
        if block:
            body['block_cjk'] = True
        for attempt in range(2):
            await self.ensure(spec.id)
            try:
                return await self._stream(sidecar, body, tenant, on_text, cancel, asked, prompt, repeat)
            except Unreachable as exc:
                reason = 'not responding' if sidecar.alive() else self._exit_reason(sidecar)
                if sidecar.state == 'ready':  # restarted by ensure() once; a restart under way is left alone
                    sidecar.state, sidecar.error = 'failed', reason
                if attempt:
                    raise ServiceError('model_unavailable', f'{spec.label} is not responding: {reason}',
                                       503) from exc

    async def _stream(self, sidecar: Sidecar, body: dict, tenant: str | None, on_text, cancel,
                      thinking: bool, prompt: int, repeat: str | None = None) -> GenerationResult:
        """thinking: the caller asked for reasoning, so the text has a <think> part even if it was skipped.
        repeat: the answer's last words the model was asked to start with; they are held back and dropped."""
        # The tenant salts the prefix cache, so one account cannot time another's cached prompts.
        headers = {**sidecar.headers, **({'X-APC-Tenant': tenant} if tenant else {})}
        reasoning = content = ''
        calls, usage, finish = [], {}, None
        cut = 0 if repeat is None else None  # characters of the content that repeat the answer; None: unknown yet

        def shown() -> str:
            nonlocal cut
            if cut is None and len(content) > len(repeat):  # enough to tell whether the model repeated it
                cut = repeated_prefix(repeat, content)
            text = content[cut:] if cut is not None else ''
            return text.lstrip() if repeat and repeat[-1:].isspace() else text

        async def read() -> None:
            nonlocal reasoning, content, usage, finish
            opened = False
            try:
                async with self._client(sidecar) as client, client.stream('POST', '/v1/chat/completions',
                                                                          json=body, headers=headers) as response:
                    opened = True
                    if response.status_code != 200:
                        raise http_error(response.status_code, (await response.aread()).decode('utf-8', 'replace'))
                    if on_text is not None:  # ChatService's relay expects a worker thread, as on the Transformers path
                        await anyio.to_thread.run_sync(on_text, '')
                    async for line in response.aiter_lines():
                        if cancel is not None and cancel.is_set():
                            break  # leaving the block closes the stream, which stops the server
                        data = line[5:].strip() if line.startswith('data:') else ''
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data) if data else None
                        except ValueError:
                            chunk = None
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get('error'):
                            raise ServiceError('generation_failed',
                                               f"The model server failed: {str(chunk['error'])[:300]}", 502)
                        usage = chunk.get('usage') or usage
                        seen = len(reasoning) + len(content)
                        for choice in chunk.get('choices') or []:
                            delta = choice.get('delta') or {}
                            # The server sends the reasoning under both names.
                            reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                            content += delta.get('content') or ''
                            calls.extend(delta.get('tool_calls') or [])
                            finish = choice.get('finish_reason') or finish
                        if on_text is not None and len(reasoning) + len(content) > seen:
                            await anyio.to_thread.run_sync(on_text, raw_text(reasoning, shown(), thinking,
                                                                             done=False))
            except httpx.TransportError as exc:
                if not opened and (isinstance(exc, httpx.ConnectError) or not sidecar.alive()):
                    raise Unreachable(str(exc)) from exc
                if not sidecar.alive():
                    raise ServiceError('model_unavailable', f'{sidecar.spec.label} stopped while answering',
                                       503) from exc
                raise ServiceError('generation_failed', f'The model server connection failed: {exc!r}', 502) from exc

        failure = None
        async with anyio.create_task_group() as group:
            if cancel is not None:
                # Stop must also end a long prompt read, before any line arrives: cancelling the read leaves the
                # stream, which closes the connection, and the server drops the request.
                group.start_soon(watch_cancel, cancel, group.cancel_scope)
            try:
                await read()
            except Exception as exc:  # raised as itself below; inside the group it would become an ExceptionGroup
                failure = exc
            group.cancel_scope.cancel()  # the watcher
        if failure is not None:
            raise failure
        if cancel is not None and cancel.is_set():
            finish = 'cancelled'
        if cut is None:  # the whole continuation was shorter than the words it was asked to repeat
            cut = repeated_prefix(repeat, content)
        content = shown()
        if thinking:
            reasoning, content = move_tool_calls(reasoning, content)
        content += ''.join(server_call(call) for call in calls)
        return GenerationResult(raw_text(reasoning, content, thinking), finish if finish in ('cancelled', 'length')
                                else 'stop', usage.get('prompt_tokens') or prompt, usage.get('completion_tokens') or 0)
