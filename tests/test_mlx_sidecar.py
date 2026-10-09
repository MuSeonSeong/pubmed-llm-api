"""The sidecar wrapper's CJK mask and mlx_vlm patch, with a fake tokenizer and fake mlx_vlm modules."""
import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace
import pytest


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's table, which Qwen's byte-level BPE vocabulary is written in."""
    bs = [*range(ord('!'), ord('~') + 1), *range(ord('¡'), ord('¬') + 1), *range(ord('®'), ord('ÿ') + 1)]
    cs, extra = bs[:], 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + extra)
            extra += 1
    return dict(zip(bs, map(chr, cs)))


def piece(raw: bytes) -> str:
    table = bytes_to_unicode()
    return ''.join(table[byte] for byte in raw)


VOCAB = [  # (token piece, blocked)
    (piece(b'hello'), False),
    (piece(' 안녕하세요'.encode()), False),
    (piece('中'.encode()), True),
    (piece(b'\xe4'), True),           # the lead byte of U+4000-U+4FFF on its own
    (piece(b'\xe4\xb8'), True),       # two of the three bytes of 一-丿
    (piece('あ'.encode()), True),
    (piece('。'.encode()), False),    # CJK punctuation stays: Korean text uses it too
    ('<|im_end|>', False),
    (piece('豈'.encode()), True),     # a compatibility ideograph
    (piece('ㅋㅋ'.encode()), False),  # Hangul compatibility jamo share the lead byte 0xE3 with kana
    (piece(b'\xea'), False),          # Hangul's own lead byte
    ('<0xE5>', True),                 # SentencePiece byte fallback
    ('▁中文', True),              # a SentencePiece word
    (None, False),                    # a hole in the vocabulary
    (piece('㐀'.encode()), True),     # CJK extension A, lead byte 0xE3
]


class Tokenizer:
    def __len__(self):
        return len(VOCAB)

    def convert_ids_to_tokens(self, ids):
        return [VOCAB[token_id][0] for token_id in ids]


def test_the_mask_covers_chinese_and_japanese_but_never_hangul():
    from app.mlx_sidecar import blocked_token_ids
    assert blocked_token_ids(Tokenizer()) == [n for n, (_, blocked) in enumerate(VOCAB) if blocked]


def test_the_mask_is_padded_or_cut_to_the_logits_width():
    import numpy as np
    from app.mlx_sidecar import BlockCJK
    mask = BlockCJK([1, 4])
    assert mask.values(3).tolist() == [0, -np.inf, 0]
    assert mask.values(6).tolist() == [0, -np.inf, 0, 0, -np.inf, 0]


def test_the_mask_is_added_to_the_logits_in_their_dtype():
    mx = pytest.importorskip('mlx.core')
    from app.mlx_sidecar import BlockCJK
    mask = BlockCJK([1, 4])
    with mx.stream(mx.cpu):  # never the GPU in tests
        logits = mx.ones((1, 6), dtype=mx.bfloat16)
        out = mask.process_last_token(7, logits)
        assert out.dtype == mx.bfloat16
        assert out.tolist() == [[1, float('-inf'), 1, 1, float('-inf'), 1]]
        assert mask(mx.array([1, 2]), mx.zeros((1, 3))).tolist() == [[0, float('-inf'), 0]]


class Wrapped:
    def __init__(self, processor):
        self.processor = processor


def fake_mlx_vlm():
    """request_normalization and generation as far as the patch touches them; a fresh class every time."""
    def build_gen_args(request, processor=None, tenant_id=None, structured_logits_processor_builder=None):
        grammar = ['grammar'] if getattr(request, 'response_format', None) else None
        return SimpleNamespace(logits_processors=grammar, tenant_id=tenant_id)

    class ResponseGenerator:
        def __init__(self, model_path='m', adapter_path=None, vision_cache=None, kv_bits=None):
            self.model_path, self.vision_cache, self.kv_bits = model_path, vision_cache, kv_bits

        def _wrap_processors_until_thinking_done(self, args, processors):
            return [Wrapped(processor) for processor in processors]
    return SimpleNamespace(_build_gen_args=build_gen_args), SimpleNamespace(ResponseGenerator=ResponseGenerator)


def fake_cli(calls):
    """mlx_vlm.server.cli: main() ends with uvicorn.run(app, host=…, port=…, …) through the module attribute."""
    uvicorn = SimpleNamespace(run=lambda app, **options: calls.append(('run', app, options)))
    cli = SimpleNamespace(uvicorn=uvicorn)
    cli.main = lambda: cli.uvicorn.run('mlx_vlm.server:app', host='0.0.0.0', port=8080, workers=1, reload=False,
                                       server_header=False)
    return cli


def test_block_cjk_requests_get_one_mask_that_also_covers_the_reasoning():
    from app.mlx_sidecar import BlockCJK, patch
    normalization, generation = fake_mlx_vlm()
    patch(normalization, generation)
    processor = SimpleNamespace(tokenizer=Tokenizer())
    assert normalization._build_gen_args(SimpleNamespace(), processor=processor).logits_processors is None
    assert normalization._build_gen_args(SimpleNamespace(block_cjk='yes'), processor=processor).logits_processors is None
    asked = normalization._build_gen_args(SimpleNamespace(block_cjk=True), processor=processor, tenant_id='user-1')
    mask, = asked.logits_processors
    assert isinstance(mask, BlockCJK) and mask.ids == blocked_ids() and asked.tenant_id == 'user-1'
    both = normalization._build_gen_args(SimpleNamespace(block_cjk=True, response_format={'type': 'json_object'}),
                                         processor=processor)
    assert both.logits_processors == ['grammar', mask]  # built once per model
    wrapped = generation.ResponseGenerator()._wrap_processors_until_thinking_done(None, both.logits_processors)
    assert wrapped[0] is mask  # active from the first reasoning token, not only after </think>
    assert isinstance(wrapped[1], Wrapped) and wrapped[1].processor == 'grammar'
    assert generation.ResponseGenerator()._wrap_processors_until_thinking_done(None, [mask]) == [mask]


def blocked_ids():
    return [n for n, (_, blocked) in enumerate(VOCAB) if blocked]


def test_a_version_without_the_hooks_refuses_to_start():
    from app.mlx_sidecar import patch
    with pytest.raises(SystemExit, match='0.7.6'):
        patch(SimpleNamespace(), SimpleNamespace(ResponseGenerator=object))
    normalization, _ = fake_mlx_vlm()
    with pytest.raises(SystemExit):
        patch(normalization, SimpleNamespace())


def test_the_sidecar_serves_only_on_its_unix_socket():
    from app.mlx_sidecar import serve_on_socket
    calls = []
    cli = fake_cli(calls)
    serve_on_socket(cli, '/tmp/pubmed-mlx-x/0.sock')
    cli.main()
    assert calls == [('run', 'mlx_vlm.server:app', {
        'uds': '/tmp/pubmed-mlx-x/0.sock', 'workers': 1, 'reload': False, 'server_header': False})]
    with pytest.raises(SystemExit, match='0.7.6'):  # a cli that binds some other way never starts
        serve_on_socket(SimpleNamespace(main=lambda: None), '/tmp/pubmed-mlx-x/0.sock')


def test_without_its_socket_the_sidecar_refuses_to_start(monkeypatch):
    import sys
    import threading
    import app.mlx_sidecar as sidecar
    monkeypatch.delenv('MLX_SIDECAR_SOCKET', raising=False)
    loaded = set(sys.modules)
    with pytest.raises(SystemExit, match='MLX_SIDECAR_SOCKET'):  # never mlx_vlm's default 0.0.0.0:8080
        sidecar.main(['--host', '127.0.0.1', '--model', 'models/qwen3.5-9b'])
    assert not [name for name in set(sys.modules) - loaded if name.startswith('mlx')]  # refused before anything
    assert not [thread for thread in threading.enumerate() if thread.name == 'parent-watch']


def test_with_accounts_the_server_caches_no_image_features():
    from app.mlx_sidecar import keep_images_private
    _, generation = fake_mlx_vlm()
    keep_images_private(generation)
    cache = object()
    assert generation.ResponseGenerator('m', vision_cache=cache, kv_bits=4).vision_cache is None
    made = generation.ResponseGenerator('m', None, cache, 4)  # passed by position, too
    assert (made.model_path, made.vision_cache, made.kv_bits) == ('m', None, 4)
    with pytest.raises(SystemExit, match='0.7.6'):
        keep_images_private(SimpleNamespace(ResponseGenerator=object))
    with pytest.raises(SystemExit):
        keep_images_private(SimpleNamespace())


def test_main_patches_mlx_vlm_as_the_app_asks_and_serves_on_the_socket(monkeypatch):
    import sys
    import app.mlx_sidecar as sidecar
    calls = []
    normalization, generation = fake_mlx_vlm()
    cli = fake_cli(calls)
    monkeypatch.setitem(sys.modules, 'mlx_vlm', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'mlx_vlm.server', SimpleNamespace(
        cli=cli, generation=generation, request_normalization=normalization))
    monkeypatch.setattr(sidecar, 'watch_parent', lambda parent: None)
    monkeypatch.setattr(sys, 'argv', ['app.mlx_sidecar'])
    monkeypatch.setenv('MLX_SIDECAR_SOCKET', '/tmp/pubmed-mlx-x/1.sock')
    monkeypatch.setenv('MLX_SIDECAR_PRIVATE', '1')
    sidecar.main(['--host', '127.0.0.1', '--model', 'models/qwen3.5-4b'])
    assert sys.argv == ['mlx_vlm.server', '--host', '127.0.0.1', '--model', 'models/qwen3.5-4b']
    assert calls == [('run', 'mlx_vlm.server:app', {
        'uds': '/tmp/pubmed-mlx-x/1.sock', 'workers': 1, 'reload': False, 'server_header': False})]
    assert generation.ResponseGenerator('m', vision_cache=object()).vision_cache is None
    asked = normalization._build_gen_args(SimpleNamespace(block_cjk=True), processor=SimpleNamespace(
        tokenizer=Tokenizer()))
    assert [type(processor).__name__ for processor in asked.logits_processors] == ['BlockCJK']  # patched as well
    monkeypatch.delenv('MLX_SIDECAR_PRIVATE')
    _, generation = fake_mlx_vlm()
    monkeypatch.setitem(sys.modules, 'mlx_vlm.server', SimpleNamespace(
        cli=fake_cli(calls), generation=generation, request_normalization=normalization))
    sidecar.main([])
    cache = object()
    assert generation.ResponseGenerator('m', vision_cache=cache).vision_cache is cache  # local mode keeps it


def test_the_pinned_mlx_vlm_still_has_the_hooks_the_patch_uses():
    found = importlib.util.find_spec('mlx_vlm')  # read as files: importing the server takes seconds
    if found is None:
        pytest.skip('mlx-vlm is installed on Apple Silicon only')
    from importlib.metadata import version
    assert version('mlx-vlm') == '0.7.6'
    server = Path(found.origin).parent / 'server'
    assert 'def _build_gen_args(' in (server / 'request_normalization.py').read_text()
    assert '_request_normalization._build_gen_args(' in (server / 'app.py').read_text()  # via the module attribute
    generation = (server / 'generation.py').read_text()
    assert 'def _wrap_processors_until_thinking_done(' in generation
    assert 'extra="allow"' in (server / 'schemas.py').read_text()  # block_cjk passes request validation
    cli = (server / 'cli.py').read_text()
    assert '\nimport uvicorn\n' in cli and cli.count('uvicorn.run(') == 1  # the one call serve_on_socket redirects
    assert 'vision_cache=None,' in generation and 'self.vision_cache is not None' in generation
    assert 'vision_cache=vision_cache,' in (server / 'app.py').read_text()  # passed by name


def test_the_sidecar_stops_once_its_parent_is_gone(monkeypatch):
    import app.mlx_sidecar as sidecar
    parents, killed = iter([100, 100, 1]), []
    monkeypatch.setattr(sidecar.os, 'getppid', lambda: next(parents))
    monkeypatch.setattr(sidecar.os, 'kill', lambda pid, sig: killed.append(sig))
    monkeypatch.setattr(sidecar.time, 'sleep', lambda seconds: None)

    def hard_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(sidecar.os, '_exit', hard_exit)
    with pytest.raises(SystemExit):
        sidecar.watch_parent(100)
    assert killed == [signal.SIGTERM]


def test_the_wrapper_imports_without_mlx():
    import subprocess
    import sys
    from app.catalog import ROOT
    code = 'import sys, app.mlx_sidecar; print(any(name.startswith("mlx") for name in sys.modules))'
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'False'
