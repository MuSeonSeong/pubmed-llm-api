"""One model at a time, with a worker-owned lock even after cancellation."""
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import gc
import threading
import anyio
import torch
from app.catalog import ModelSpec, model_weight_files
from app.settings import Settings, ServiceError

@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def select_device(requested: str, *, cuda_available: bool, mps_available: bool) -> str:
    available = {'cuda': cuda_available, 'mps': mps_available, 'cpu': True}
    if requested == 'auto':
        return next(d for d in ['cuda', 'mps', 'cpu'] if available[d])
    if not available.get(requested, False):
        raise ServiceError('device_unavailable', f'Device {requested} is unavailable', 503)
    return requested

class TextCallbackStreamer:
    """Report the cumulative decoded completion, starting with '' once the prompt is accepted."""
    def __init__(self, tokenizer, on_text: Callable[[str], None]):
        self.tokenizer, self.on_text = tokenizer, on_text
        self.ids, self.prompt_seen = [], False

    def put(self, value) -> None:
        if not self.prompt_seen:  # generate() first passes the prompt ids
            self.prompt_seen = True
            self.on_text('')
            return
        self.ids.extend(value.reshape(-1).tolist())
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        if not text.endswith('\ufffd'):  # wait for the rest of a multi-byte character
            self.on_text(text)

    def end(self) -> None:
        pass


class CancelCriteria:
    def __init__(self, cancel: threading.Event):
        self.cancel = cancel

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.cancel.is_set(), dtype=torch.bool, device=input_ids.device)


class ModelManager:
    def __init__(self, settings: Settings, catalog: dict[str, ModelSpec]):
        self.settings, self.catalog = settings, catalog
        self.device = select_device(settings.device, cuda_available=torch.cuda.is_available(),
                                    mps_available=torch.backends.mps.is_available())
        self._lock = threading.Lock()
        self._model = self._tokenizer = None
        self._loaded_id = None

    def status(self) -> dict:
        return {'device': self.device, 'loaded_model': self._loaded_id, 'busy': self._lock.locked(),
                'model_files': {name: self._files_available(self.settings.models_dir / spec.directory)
                                for name, spec in self.catalog.items()}}

    @staticmethod
    def _files_available(path) -> bool:
        if not all((path / name).is_file() for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']):
            return False
        try:
            model_weight_files(path)
            return True
        except (OSError, ValueError, KeyError):
            return False

    def _release(self) -> None:
        self._model = self._tokenizer = None
        self._loaded_id = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()

    def close(self) -> None:
        with self._lock:
            self._release()

    def _load(self, spec: ModelSpec):
        from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration
        path = self.settings.models_dir / spec.directory
        if not self._files_available(path):
            raise ServiceError('model_missing', 'Run: uv run python scripts/prepare_models.py', 503)
        dtype = torch.float32 if self.device == 'cpu' else torch.float16
        if self.device == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        loader = Qwen3_5ForConditionalGeneration if spec.loader == 'qwen3_5' else AutoModelForCausalLM
        model = loader.from_pretrained(path, dtype=dtype, device_map='cpu' if self.device == 'mps' else self.device,
                                       attn_implementation='sdpa', local_files_only=True)
        if self.device == 'mps':
            # Direct safetensors-to-MPS placement crashes with this runtime combination.
            model = model.to('mps')
        model.eval()
        return model, tokenizer

    async def generate(self, model_id: str, messages: list[dict], tools: list[dict],
                       max_tokens: int, temperature: float, *, on_text: Callable[[str], None] | None = None,
                       cancel: threading.Event | None = None, thinking: bool = False,
                       continue_final: bool = False) -> GenerationResult:
        """on_text runs on the worker thread with the cumulative raw completion text."""
        if model_id not in self.catalog:
            raise ServiceError('unknown_model', f'Unknown model: {model_id}', 422)
        return await anyio.to_thread.run_sync(partial(self._generate_sync, model_id, messages, tools,
                                                     max_tokens, temperature, on_text, cancel, thinking,
                                                     continue_final),
                                              abandon_on_cancel=False)

    def _generate_sync(self, model_id, messages, tools, max_tokens, temperature, on_text=None, cancel=None,
                       thinking=False, continue_final=False):
        with self._lock:
            if cancel is not None and cancel.is_set():
                return GenerationResult('', 'cancelled', 0, 0)
            try:
                if self._loaded_id != model_id:
                    self._release()
                    self._model, self._tokenizer = self._load(self.catalog[model_id])
                    self._loaded_id = model_id
                tokenizer, model = self._tokenizer, self._model
                template = ({'continue_final_message': True, 'add_generation_prompt': False} if continue_final
                            else {'add_generation_prompt': True})
                inputs = tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=True,
                    return_dict=True, return_tensors='pt', enable_thinking=thinking, **template)
                count = inputs['input_ids'].shape[-1]
                if count + max_tokens > self.settings.max_context_tokens:
                    raise ServiceError('context_overflow', 'Input plus requested output exceeds context budget',
                                       413, {'prompt_tokens': count, 'max_tokens': max_tokens,
                                             'limit': self.settings.max_context_tokens})
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                              pad_token_id=tokenizer.eos_token_id)
                if temperature > 0:
                    kwargs['temperature'] = temperature
                if on_text is not None:
                    kwargs['streamer'] = TextCallbackStreamer(tokenizer, on_text)
                if cancel is not None:
                    from transformers import StoppingCriteriaList
                    kwargs['stopping_criteria'] = StoppingCriteriaList([CancelCriteria(cancel)])
                with torch.inference_mode():
                    outputs = model.generate(**inputs, **kwargs)
                generated = outputs[0, count:].tolist()
                eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', tokenizer.eos_token_id)
                eos = eos if isinstance(eos, list) else [eos]
                reason = 'stop' if generated and generated[-1] in eos else 'length'
                if cancel is not None and cancel.is_set() and reason == 'length':
                    reason = 'cancelled'
                return GenerationResult(tokenizer.decode(generated, skip_special_tokens=False),
                                        reason, count, len(generated))
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError('generation_failed', str(exc), 500) from exc
