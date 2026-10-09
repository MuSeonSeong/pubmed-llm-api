"""Environment configuration and errors shared across service boundaries."""
import importlib.util
from pathlib import Path
import platform
import sys
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.catalog import ROOT

# PRELOAD_MODELS was a boolean (pydantic's spellings): true preloaded every MLX model, false none.
LEGACY_PRELOAD = {**dict.fromkeys(['true', '1', 'yes', 'on', 't', 'y'], 'all'),
                  **dict.fromkeys(['false', '0', 'no', 'off', 'f', 'n'], 'none')}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto'
    models_dir: Path = ROOT / 'models'
    # Catalog ids to serve, comma separated; the first is the default model. Empty: every MLX model on
    # Apple Silicon with mlx-vlm installed, otherwise every Transformers model.
    models: str = ''
    # Attachments, images, PubMed abstracts and reasoning all share this budget (Transformers models;
    # MLX models set their own context_tokens in models.json).
    max_context_tokens: int = Field(default=8192, ge=1, le=8192)
    # MLX models started and warmed with the app, kept loaded: the default model, all or none; the others start
    # on their first question. On a 16 GB Mac a second resident model got the idle default swapped out.
    preload_models: Literal['default', 'all', 'none'] = 'default'
    # An MLX model that was not preloaded is stopped after this many minutes without a request, giving its
    # memory back; its next question starts it again. 0: never.
    mlx_idle_minutes: int = Field(default=10, ge=0)
    # Disk tier of the MLX prefix cache, in GB for all models together, kept in DATA_DIR/kv-cache. 0 (the default)
    # keeps the cache in memory only: the disk tier stores prompts' token ids, which deleting a chat or an account
    # does not remove.
    kv_cache_gb: float = Field(default=0, ge=0, le=1024)
    # Qwen3.5's recurrent layers resume a cached prefix only from checkpoints stored every this many tokens,
    # keeping this many per prompt; the mlx-vlm defaults (2048, 2) never reuse a 1.6k-token PubMed prompt.
    mlx_checkpoint_interval: int = Field(default=512, ge=16, le=65536)
    mlx_checkpoint_entries: int = Field(default=8, ge=1, le=256)
    pubmed_mcp_url: str = 'https://pubmed.caseyjhand.com/mcp'
    # Accounts, chats and uploaded images; nothing is written here until it is first needed.
    data_dir: Path = ROOT / 'data'
    # Off: a single implicit 'local' user and no sign-in. On: every API call needs a session.
    auth_enabled: bool = False
    session_days: int = Field(default=30, ge=1, le=365)
    # Per user, chat JSON and uploaded images together.
    user_storage_mb: int = Field(default=500, ge=1, le=100000)

    @field_validator('models_dir', 'data_dir')
    @classmethod
    def portable_path(cls, path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    @field_validator('preload_models', mode='before')
    @classmethod
    def preload_mode(cls, value):
        if isinstance(value, bool | int | str):
            text = str(value).strip().lower()
            return LEGACY_PRELOAD.get(text, text)
        return value


def mlx_available() -> bool:
    """MLX models need Apple Silicon and mlx-vlm, which uv.lock installs only there; found without importing it."""
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        return False
    return importlib.util.find_spec('mlx_vlm') is not None


def enabled_models(settings: Settings, catalog: dict, mlx: bool | None = None) -> list[str]:
    """The catalog ids this server answers with, default model first; ValueError for a wrong MODELS."""
    mlx = mlx_available() if mlx is None else mlx
    ids = list(dict.fromkeys(part.strip() for part in settings.models.split(',') if part.strip()))
    if ids:
        unknown = [model_id for model_id in ids if model_id not in catalog]
        if unknown:
            raise ValueError(f"MODELS names unknown models: {', '.join(unknown)} (known: {', '.join(catalog)})")
        if not mlx and any(catalog[model_id].loader == 'mlx' for model_id in ids):
            raise ValueError('MODELS names MLX models, which need mlx-vlm on Apple Silicon (macOS arm64)')
    else:
        ids = [model_id for model_id, spec in catalog.items() if (spec.loader == 'mlx') == mlx]
        if not ids:
            raise ValueError('No models to serve; set MODELS')
    return ids


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: dict | None = None,
                 headers: dict[str, str] | None = None):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code
        self.details = details or {}
        self.headers = headers or {}
