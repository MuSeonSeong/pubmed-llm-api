"""Pinned model identifiers and portable local paths."""
from dataclasses import dataclass, field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# causal_lm and qwen3_5 run on Transformers in this process; mlx runs in an mlx_vlm.server sidecar.
LOADERS = frozenset({'causal_lm', 'qwen3_5', 'mlx'})
SAMPLING_KEYS = frozenset({'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'})


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo_id: str
    revision: str
    directory: str
    loader: str
    label: str = ''  # name shown in the UI; the id when empty
    summary: str = ''  # short Korean hint shown next to the label
    vision: bool | None = None  # None: Qwen3.5 checkpoints (loader qwen3_5) include a vision encoder, Qwen3 does not
    reasoning: bool = True
    context_tokens: int | None = None  # None: Settings.max_context_tokens
    # The model card's recommended sampling for direct answers and for answers with reasoning.
    sampling: dict = field(default_factory=dict, hash=False)
    thinking_sampling: dict = field(default_factory=dict, hash=False)
    # app: the raw <tool_call> markup reaches app/chat_service.py, which parses, repairs and corrects it.
    tool_parsing: str = 'app'

    def __post_init__(self):
        if self.vision is None:
            object.__setattr__(self, 'vision', self.loader == 'qwen3_5')
        if not self.label:
            object.__setattr__(self, 'label', self.id)


def load_catalog() -> dict[str, ModelSpec]:
    entries = json.loads((ROOT / "models.json").read_text())
    catalog = {item["id"]: ModelSpec(**item) for item in entries}
    for spec in catalog.values():
        if (Path(spec.directory).name != spec.directory or spec.loader not in LOADERS
                or spec.tool_parsing not in ('app', 'server')
                or not set(spec.sampling) | set(spec.thinking_sampling) <= SAMPLING_KEYS
                or (spec.context_tokens is not None and spec.context_tokens < 1)):
            raise ValueError("Invalid model manifest")
    return catalog


def model_weight_files(path: Path) -> list[Path]:
    """Use the same checkpoint entry points as the Transformers loader."""
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        data = json.loads(index.read_text())
        mapping = data.get('weight_map') if isinstance(data, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'Invalid or empty weight index in {path}')
        names = list(mapping.values())
        if any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts for name in names):
            raise ValueError(f'Invalid shard path in {path}')
        files = [path / name for name in sorted(set(names))]
    else:
        files = [path / 'model.safetensors']
    if not all(file.is_file() and file.stat().st_size > 0 for file in files):
        raise ValueError(f'Missing recognized weights or shard index in {path}')
    return files


def files_ready(path: Path) -> bool:
    """Whether scripts/prepare_models.py has left a complete model in `path`."""
    if not all((path / name).is_file() for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']):
        return False
    try:
        model_weight_files(path)
        return True
    except (OSError, ValueError, KeyError):
        return False
