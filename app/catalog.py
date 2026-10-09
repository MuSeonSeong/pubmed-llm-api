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
