"""Pinned model identifiers and portable local paths."""
from dataclasses import dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo_id: str
    revision: str
    directory: str
    loader: str


def load_catalog() -> dict[str, ModelSpec]:
    entries = json.loads((ROOT / "models.json").read_text())
    catalog = {item["id"]: ModelSpec(**item) for item in entries}
    for spec in catalog.values():
        if Path(spec.directory).name != spec.directory or spec.loader not in {"causal_lm", "qwen3_5"}:
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
