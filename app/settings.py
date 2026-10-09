"""Environment configuration and errors shared across service boundaries."""
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.catalog import ROOT

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto'
    models_dir: Path = ROOT / 'models'
    # Attachments, images, PubMed abstracts and reasoning all share this budget.
    max_context_tokens: int = Field(default=8192, ge=1, le=8192)
    pubmed_mcp_url: str = 'https://pubmed.caseyjhand.com/mcp'

    @field_validator('models_dir')
    @classmethod
    def portable_path(cls, path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code
        self.details = details or {}
