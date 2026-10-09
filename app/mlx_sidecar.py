"""Run mlx_vlm's server for one model, as app/mlx_runtime.py starts it.

  MLX_SIDECAR_SOCKET=<private dir>/0.sock python -m app.mlx_sidecar --host 127.0.0.1 --model models/qwen3.5-9b ...

It serves only on that Unix socket, never on a TCP port. It exits when the app that started it dies, so a
crashed or killed app never leaves a model holding GPU memory. It adds one request field, block_cjk: Qwen3.5
writes stray Chinese or Japanese words into Korean answers (2-3 of 13 on the 9B, whatever the sampling), which
a logits mask rules out. With MLX_SIDECAR_PRIVATE=1 (accounts) it caches no image features.
"""
import functools
import inspect
import logging
import os
import re
import signal
import sys
import threading
import time

log = logging.getLogger('mlx_vlm.server')
PARENT_POLL_SECONDS = 1
# Set by app/mlx_runtime.py: the app's pid, the socket to serve on and whether accounts share the server.
PARENT_ENV, SOCKET_ENV, PRIVATE_ENV = 'MLX_SIDECAR_PARENT_PID', 'MLX_SIDECAR_SOCKET', 'MLX_SIDECAR_PRIVATE'
# Kana, CJK extension A, unified and compatibility ideographs; Hangul (and CJK punctuation) stay allowed.
CJK = re.compile('[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
# UTF-8 lead bytes of U+4000-U+9FFF, never used by Hangul (0xEA-0xED): blocking every token that holds one
# keeps the model from spelling a Chinese character byte by byte.
CJK_LEAD_BYTES = frozenset(range(0xE4, 0xEA))
SPIECE_BYTE = re.compile(r'<0x([0-9A-Fa-f]{2})>')


def watch_parent(parent: int, interval: float = PARENT_POLL_SECONDS) -> None:
    """Stop once this process is re-parented, which means the app that started it is gone."""
    while os.getppid() == parent:
        time.sleep(interval)
    os.kill(os.getpid(), signal.SIGTERM)  # uvicorn shuts down; if that hangs, exit anyway
    time.sleep(10)
    os._exit(1)


def byte_decoder() -> dict[str, int]:
    """The inverse of GPT-2's bytes_to_unicode, which byte-level BPE vocabularies (Qwen's) are written in."""
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]  # bytes shown as themselves
    decoder, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            decoder[chr(byte)] = byte
        else:
            decoder[chr(256 + extra)] = byte
            extra += 1
    return decoder


def token_bytes(piece: str, decoder: dict[str, int]) -> bytes:
    match = SPIECE_BYTE.fullmatch(piece)  # SentencePiece byte fallback
    if match:
        return bytes([int(match.group(1), 16)])
    if all(char in decoder for char in piece):
        return bytes(decoder[char] for char in piece)
    return piece.replace('\u2581', ' ').encode()  # SentencePiece marks spaces with U+2581


def blocked_token_ids(tokenizer) -> list[int]:
    """Tokens that write a Chinese or Japanese character, or hold a byte that starts a Chinese one."""
    decoder = byte_decoder()
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    blocked = []
    for token_id, piece in enumerate(pieces):
        if not piece:
            continue
        raw = token_bytes(piece, decoder)
        if CJK_LEAD_BYTES.intersection(raw) or CJK.search(raw.decode('utf-8', 'ignore')):
            blocked.append(token_id)
    return blocked


class BlockCJK:
    """Logits processor adding -inf at the blocked tokens; it keeps no state, so one serves every request."""

    def __init__(self, ids: list[int]):
        self.ids = ids
        self._masks = {}

    def values(self, width: int):
        """The additive mask, padded or cut to the logits' width (the embedding may be wider than the vocab)."""
        import numpy as np
        mask = np.zeros(width, dtype=np.float32)
        mask[[token_id for token_id in self.ids if token_id < width]] = -np.inf
        return mask

    def __call__(self, tokens, logits):
        key = (logits.shape[-1], logits.dtype)
        if key not in self._masks:
            import mlx.core as mx
            self._masks[key] = mx.array(self.values(key[0])).astype(key[1])
        return logits + self._masks[key]

    def process_last_token(self, token, logits):  # mlx-vlm's batch generator calls this when it exists
        return self(None, logits)


_masks: list[tuple[object, BlockCJK]] = []  # (tokenizer, its mask); a sidecar serves one model
_masks_lock = threading.Lock()


def cjk_processor(processor) -> BlockCJK:
    """The model's mask, built once from its tokenizer (about a second for Qwen3.5's 248k tokens)."""
    tokenizer = getattr(processor, 'tokenizer', processor)
    with _masks_lock:
        for known, mask in _masks:
            if known is tokenizer:
                return mask
        started = time.monotonic()
        mask = BlockCJK(blocked_token_ids(tokenizer))
        _masks.append((tokenizer, mask))
        log.info('block_cjk: %d tokens masked in %.1f s', len(mask.ids), time.monotonic() - started)
        return mask


def patch(request_normalization, generation) -> None:
    """Teach mlx_vlm's request handling block_cjk; refuse to run on a version without these hooks."""
    build = getattr(request_normalization, '_build_gen_args', None)
    wrap = getattr(getattr(generation, 'ResponseGenerator', None), '_wrap_processors_until_thinking_done', None)
    if build is None or wrap is None:
        raise SystemExit('app.mlx_sidecar needs mlx-vlm 0.7.6: request_normalization._build_gen_args or '
                         'ResponseGenerator._wrap_processors_until_thinking_done is missing')

    def build_gen_args(request, *args, **kwargs):
        gen_args = build(request, *args, **kwargs)
        processor = kwargs.get('processor', args[0] if args else None)
        # Request models accept extra fields, so block_cjk arrives as an attribute.
        if getattr(request, 'block_cjk', False) is True and processor is not None:
            gen_args.logits_processors = [*(gen_args.logits_processors or []), cjk_processor(processor)]
        return gen_args

    def wrap_until_thinking_done(self, args, processors):
        # mlx-vlm holds request processors back until </think> (they are meant for JSON grammars), but
        # Qwen sometimes reasons in Chinese, which must not happen either.
        kept = [processor for processor in processors if isinstance(processor, BlockCJK)]
        rest = [processor for processor in processors if not isinstance(processor, BlockCJK)]
        return kept + (wrap(self, args, rest) if rest else [])

    request_normalization._build_gen_args = build_gen_args  # app.py calls it through the module attribute
    generation.ResponseGenerator._wrap_processors_until_thinking_done = wrap_until_thinking_done


def keep_images_private(generation) -> None:
    """Cache no image features: mlx-vlm keys them by the image's content, not by account, so with accounts the
    timing of an answer would tell whether another account had sent the same image."""
    generator = getattr(generation, 'ResponseGenerator', None)
    try:
        signature = inspect.signature(generator.__init__)
    except (AttributeError, TypeError, ValueError):
        signature = None
    if signature is None or 'vision_cache' not in signature.parameters:
        raise SystemExit('app.mlx_sidecar needs mlx-vlm 0.7.6: ResponseGenerator takes no vision_cache')
    init = generator.__init__

    @functools.wraps(init)
    def without_vision_cache(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.arguments['vision_cache'] = None  # _gpu_embed then encodes every image afresh
        init(*bound.args, **bound.kwargs)
    generator.__init__ = without_vision_cache


def serve_on_socket(cli, path: str) -> None:
    """Make mlx_vlm's cli serve on the Unix socket at path: it ends with uvicorn.run(app, host=…, port=…, …),
    called through the module attribute."""
    uvicorn = getattr(cli, 'uvicorn', None)
    run = getattr(uvicorn, 'run', None)
    if run is None:
        raise SystemExit('app.mlx_sidecar needs mlx-vlm 0.7.6: mlx_vlm.server.cli does not call uvicorn.run')

    def run_on_socket(app, **options):
        options.pop('host', None)
        options.pop('port', None)
        return run(app, uds=path, **options)
    uvicorn.run = run_on_socket


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    path = os.environ.get(SOCKET_ENV)
    if not path:  # never mlx_vlm's default of every address on port 8080
        raise SystemExit(f'app.mlx_sidecar serves only on the Unix socket in {SOCKET_ENV}, which is not set')
    # Before the slow import, so an app that dies meanwhile still takes this process with it; the pid comes
    # from the app, in case it was gone before this line ran.
    parent = int(os.environ.get(PARENT_ENV) or os.getppid())
    threading.Thread(target=watch_parent, args=(parent,), name='parent-watch', daemon=True).start()
    from mlx_vlm.server import cli, generation, request_normalization
    patch(request_normalization, generation)
    if os.environ.get(PRIVATE_ENV) == '1':
        keep_images_private(generation)
    serve_on_socket(cli, path)
    sys.argv = ['mlx_vlm.server', *argv]
    cli.main()


if __name__ == '__main__':
    main()
