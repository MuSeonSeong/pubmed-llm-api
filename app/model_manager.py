"""Every enabled model: MLX models through their sidecars (app/mlx_runtime.py), Transformers models here,
one at a time, with a worker-owned lock even after cancellation."""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
import gc
import threading
import anyio
import torch
from app.catalog import ModelSpec, files_ready
from app.images import IMAGE_TOKEN, MODEL_MAX_PIXELS, VISION_TOKENS, decode_image
from app.settings import Settings, ServiceError, enabled_models

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

# Reasoning tokens each effort may spend before reasoning is closed, for both runtimes. Qwen3.5-9B on MLX thinks
# ~700-1000 tokens about a small logic puzzle at ~20 tok/s, so these cap the wait near 25 s, 100 s, 3.5 and 7 min.
THINKING_BUDGETS = {'none': 0, 'light': 512, 'medium': 2048, 'high': 4096, 'max': 8192}
THINKING_CLOSE = ('\n\nConsidering the limited time by the user, I have to give the solution based on the '
                  'thinking directly now.\n</think>\n\n')
MIN_OUTPUT_TOKENS = 64  # below this an answer (or reasoning) is not worth starting
EMPTY_REASONING = '<think></think>'
TRANSFORMERS_TEMPERATURE = 0.2  # when a request leaves the temperature to the model


def plan_output(limit: int, prompt_tokens: int, max_tokens: int, thinking_budget: int,
                strict: bool) -> tuple[int, int]:
    """The answer length and reasoning budget that fit in the context the prompt leaves.

    Strict callers can still drop old turns, so they get the whole answer or a 413; otherwise the answer
    shrinks to the room left and only below MIN_OUTPUT_TOKENS is it refused. Reasoning gets what the answer
    leaves, and nothing when that is below MIN_OUTPUT_TOKENS.
    """
    room = limit - prompt_tokens
    if room < (max_tokens if strict else min(max_tokens, MIN_OUTPUT_TOKENS)):
        raise ServiceError('context_overflow', 'Input plus requested output exceeds context budget', 413,
                           {'prompt_tokens': prompt_tokens, 'max_tokens': max_tokens, 'limit': limit})
    answer = min(max_tokens, room)
    budget = min(thinking_budget, room - answer)
    return answer, (budget if budget >= MIN_OUTPUT_TOKENS else 0)


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
    def __init__(self, settings: Settings, catalog: dict[str, ModelSpec], pool=None):
        self.settings, self.catalog = settings, catalog
        self.enabled = enabled_models(settings, catalog)
        self.default_model = self.enabled[0]
        mlx = [catalog[model_id] for model_id in self.enabled if catalog[model_id].loader == 'mlx']
        if pool is None and mlx:
            from app.mlx_runtime import MLXPool  # imported here because it builds on this module
            pool = MLXPool(settings, mlx)
        self.pool = pool
        self.device = select_device(settings.device, cuda_available=torch.cuda.is_available(),
                                    mps_available=torch.backends.mps.is_available())
        self._lock = threading.Lock()
        self._model = self._tokenizer = None
        self._loaded_id = None
        self._vision = None  # (model id, image processor), loaded on the first image

    def status(self) -> dict:
        """Today's Transformers fields plus `models`: the state of every enabled model."""
        files = {name: files_ready(self.settings.models_dir / spec.directory) for name, spec in self.catalog.items()}
        sidecars = self.pool.status() if self.pool is not None else {}
        models = {model_id: sidecars.get(model_id) or {
            'state': 'ready' if files[model_id] else 'missing', 'runtime': 'transformers',
            'label': self.catalog[model_id].label, 'error': None} for model_id in self.enabled}
        return {'device': self.device, 'loaded_model': self._loaded_id, 'busy': self._lock.locked(),
                'max_context_tokens': self.settings.max_context_tokens, 'model_files': files, 'models': models}

    async def start(self, prefix: Callable[[], Awaitable[tuple[str, list[dict]]]] | None = None) -> None:
        """Own the MLX models' starts until cancelled at shutdown, starting the preloaded ones now; Transformers
        models load on first use.

        prefix gives the system prompt and tool schemas PubMed questions start with, cached after each start.
        """
        if self.pool is not None:
            await self.pool.run(prefix, self.preload_ids())

    def preload_ids(self) -> list[str]:
        """The MLX models PRELOAD_MODELS starts with the app and keeps loaded, default first: the default model
        (when it is an MLX model), all of them or none."""
        mode = self.settings.preload_models
        return [model_id for model_id in self.enabled if self.catalog[model_id].loader == 'mlx'
                and (mode == 'all' or mode == 'default' and model_id == self.default_model)]

    def _release(self) -> None:
        self._model = self._tokenizer = None
        self._loaded_id = None
        self._vision = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
        with self._lock:
            self._release()

    def _load(self, spec: ModelSpec):
        from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3_5ForConditionalGeneration
        path = self.settings.models_dir / spec.directory
        if not files_ready(path):
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

    def _image_processor(self, spec: ModelSpec):
        if self._vision is None or self._vision[0] != spec.id:
            # The PIL variant avoids the torchvision dependency of the default (fast) processor.
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
            # The size bounds are pixel counts: at most MODEL_MAX_PIXELS per image.
            processor = Qwen2VLImageProcessorPil.from_pretrained(
                self.settings.models_dir / spec.directory, local_files_only=True,
                size={'shortest_edge': 56 * 56, 'longest_edge': MODEL_MAX_PIXELS})
            self._vision = (spec.id, processor)
        return self._vision[1]

    @staticmethod
    def _decode_images(spec: ModelSpec, messages) -> tuple[list, list]:
        """Decode images and swap them for template placeholders; runs before any model is loaded."""
        images, templated = [], []
        for message in messages:
            if not isinstance(message.get('content'), list):
                templated.append(message)
                continue
            parts = []
            for part in message['content']:
                if part['type'] == 'text':
                    text = part['text']
                    for token in VISION_TOKENS:
                        text = text.replace(token, '')
                    parts.append({'type': 'text', 'text': text})
                else:
                    images.append(decode_image(part['image_url']['url']))
                    parts.append({'type': 'image'})
            if all(part['type'] == 'text' for part in parts):
                # Text-only templates (Qwen3) render list content as empty, so send plain text.
                templated.append({**message, 'content': '\n\n'.join(part['text'] for part in parts)})
            else:
                templated.append({**message, 'content': parts})
        if images and not spec.vision:
            raise ServiceError('vision_unsupported', f'{spec.id} cannot read images', 422)
        return images, templated

    def _prepare_inputs(self, spec: ModelSpec, tokenizer, images, templated, tools, thinking, template) -> dict:
        """Tokenize the conversation; images become placeholder tokens plus pixel tensors."""
        options = dict(tools=tools or None, enable_thinking=thinking, **template)
        if not images:
            return tokenizer.apply_chat_template(templated, tokenize=True, return_dict=True,
                                                 return_tensors='pt', **options)
        processor = self._image_processor(spec)
        features = processor(images=images, return_tensors='pt')
        text = tokenizer.apply_chat_template(templated, tokenize=False, **options)
        pieces = text.split(IMAGE_TOKEN)
        if len(pieces) - 1 != len(images):
            raise ServiceError('invalid_image', 'Image placeholders do not match the images', 422)
        merge = processor.merge_size ** 2
        text = pieces[0] + ''.join(IMAGE_TOKEN * (int(grid.prod()) // merge) + piece
                                   for grid, piece in zip(features['image_grid_thw'], pieces[1:]))
        encoded = tokenizer([text], return_tensors='pt', add_special_tokens=False)
        # Marks image tokens (type 1) so Qwen3.5 gives them 2D rotary positions; without it they
        # silently get plain text positions and lose their spatial layout.
        image_mask = encoded['input_ids'] == tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'],
                'mm_token_type_ids': image_mask.long(),
                'pixel_values': features['pixel_values'], 'image_grid_thw': features['image_grid_thw']}

    @staticmethod
    def _generate_with_thinking_budget(model, tokenizer, inputs, kwargs, count, budget, eos, cancel):
        """Budget forcing: if reasoning runs past its budget, close it and let the model answer.

        Small models can keep second-guessing until no tokens are left for the answer.
        """
        total = kwargs['max_new_tokens']  # the reasoning budget plus the answer
        outputs = model.generate(**inputs, **{**kwargs, 'max_new_tokens': budget})
        generated = outputs[0, count:].tolist()
        if (generated and generated[-1] in eos) or (cancel and cancel.is_set()):
            return outputs
        # Still reasoning: close it. Already answering: just keep going with the rest of the budget.
        closed = '</think>' in tokenizer.decode(generated, skip_special_tokens=False)
        close_ids = [] if closed else tokenizer(THINKING_CLOSE, add_special_tokens=False)['input_ids']
        forced = torch.tensor([close_ids], dtype=outputs.dtype, device=outputs.device)
        # The closing words count against the total, which never exceeds the budget plus the answer.
        remaining = total - budget - len(close_ids)
        if remaining < 1:
            return outputs
        prefix = torch.cat([outputs, forced], dim=1)
        streamer = kwargs.get('streamer')
        if streamer is not None:  # show any closing words, then treat the next prompt as already seen
            streamer.ids.extend(forced[0].tolist())
            streamer.on_text(tokenizer.decode(streamer.ids, skip_special_tokens=False))
            streamer.prompt_seen = False
        extended = {}
        for name, value in inputs.items():
            if name == 'input_ids':
                extended[name] = prefix
            elif torch.is_tensor(value) and value.dim() == 2 and value.shape[-1] == count:
                # attention_mask and mm_token_type_ids grow with plain text tokens
                fill = torch.ones if name == 'attention_mask' else torch.zeros
                pad = fill((value.shape[0], prefix.shape[-1] - count), dtype=value.dtype, device=value.device)
                extended[name] = torch.cat([value, pad], dim=1)
            else:
                extended[name] = value
        return model.generate(**extended, **{**kwargs, 'max_new_tokens': remaining})

    async def generate(self, model_id: str, messages: list[dict], tools: list[dict], max_tokens: int,
                       temperature: float | None = None, *, on_text: Callable[[str], None] | None = None,
                       cancel: threading.Event | None = None, effort: str = 'none', continue_final: bool = False,
                       strict_budget: bool = False, tenant: str | None = None) -> GenerationResult:
        """on_text runs on a worker thread with the cumulative raw completion text.

        strict_budget refuses (413) instead of shortening the answer, so callers can trim history first.
        tenant keeps one account's cached prompts apart from the others' (MLX models).
        """
        spec = self.catalog.get(model_id)
        if spec is None:
            raise ServiceError('unknown_model', f'Unknown model: {model_id}', 422)
        effort = effort if spec.reasoning else 'none'
        if spec.loader == 'mlx':
            if self.pool is None or model_id not in self.pool:
                raise ServiceError('unknown_model', f'{model_id} is not enabled on this server', 422)
            return await self.pool.generate(spec, messages, tools, max_tokens, temperature, on_text=on_text,
                                            cancel=cancel, effort=effort, continue_final=continue_final,
                                            strict_budget=strict_budget, tenant=tenant)
        return await anyio.to_thread.run_sync(partial(self._generate_sync, model_id, messages, tools,
                                                     max_tokens, temperature, on_text, cancel, effort,
                                                     continue_final, strict_budget),
                                              abandon_on_cancel=False)

    def _generate_sync(self, model_id, messages, tools, max_tokens, temperature, on_text=None, cancel=None,
                       effort='none', continue_final=False, strict_budget=False):
        spec = self.catalog[model_id]
        thinking = effort != 'none' and not continue_final
        temperature = TRANSFORMERS_TEMPERATURE if temperature is None else temperature
        with self._lock:
            if cancel is not None and cancel.is_set():
                return GenerationResult('', 'cancelled', 0, 0)
            try:
                images, templated = self._decode_images(spec, messages)
                if self._loaded_id != model_id:
                    self._release()
                    self._model, self._tokenizer = self._load(spec)
                    self._loaded_id = model_id
                tokenizer, model = self._tokenizer, self._model
                template = ({'continue_final_message': True, 'add_generation_prompt': False} if continue_final
                            else {'add_generation_prompt': True})
                limit = spec.context_tokens or self.settings.max_context_tokens
                inputs = self._prepare_inputs(spec, tokenizer, images, templated, tools, thinking, template)
                count = inputs['input_ids'].shape[-1]
                # Shorten the answer rather than refuse it; finish_reason 'length' lets clients continue.
                answer, budget = plan_output(limit, count, max_tokens, THINKING_BUDGETS[effort] if thinking else 0,
                                             strict_budget)
                skipped = ''
                if thinking and not budget:
                    # No room left to reason: answer directly, behind an empty reasoning so callers that asked
                    # for reasoning still find where the answer starts.
                    thinking, skipped = False, EMPTY_REASONING
                    inputs = self._prepare_inputs(spec, tokenizer, images, templated, tools, False, template)
                    count = inputs['input_ids'].shape[-1]
                    answer, _ = plan_output(limit, count, answer, 0, False)
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(getattr(model, 'dtype', torch.float32))
                kwargs = dict(max_new_tokens=answer + budget, do_sample=temperature > 0,
                              pad_token_id=tokenizer.eos_token_id)
                if temperature > 0:
                    kwargs['temperature'] = temperature
                if on_text is not None:
                    kwargs['streamer'] = TextCallbackStreamer(tokenizer, (lambda text: on_text(skipped + text))
                                                              if skipped else on_text)
                if cancel is not None:
                    from transformers import StoppingCriteriaList
                    kwargs['stopping_criteria'] = StoppingCriteriaList([CancelCriteria(cancel)])
                eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', tokenizer.eos_token_id)
                eos = eos if isinstance(eos, list) else [eos]
                with torch.inference_mode():
                    if thinking:
                        outputs = self._generate_with_thinking_budget(model, tokenizer, inputs, kwargs, count,
                                                                      budget, eos, cancel)
                    else:
                        outputs = model.generate(**inputs, **kwargs)
                generated = outputs[0, count:].tolist()
                reason = 'stop' if generated and generated[-1] in eos else 'length'
                if cancel is not None and cancel.is_set() and reason == 'length':
                    reason = 'cancelled'
                return GenerationResult(skipped + tokenizer.decode(generated, skip_special_tokens=False),
                                        reason, count, len(generated))
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError('generation_failed', str(exc), 500) from exc
