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

class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int, details: dict | None = None,
                 headers: dict[str, str] | None = None):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code
        self.details = details or {}
        self.headers = headers or {}
