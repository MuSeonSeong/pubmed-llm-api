"""Public chat contract; stream=true switches the response to server-sent events."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, FiniteFloat, model_validator

class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'qwen3.5-2b'
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: StrictInt = Field(default=512, ge=1, le=2048)
    temperature: FiniteFloat = Field(default=0.2, ge=0, le=2, strict=True)
    use_pubmed: bool = True
    stream: bool = False
    # Drop the oldest turns instead of failing when the conversation exceeds the context budget.
    truncate_history: bool = False

    @model_validator(mode='after')
    def require_user_query(self):
        if not any(message.role == 'user' for message in self.messages):
            raise ValueError('At least one user message is required')
        return self

class ToolTrace(BaseModel):
    name: str
    arguments: dict
    result: dict | None = None
    error: str | None = None

class AssistantMessage(BaseModel):
    role: Literal['assistant'] = 'assistant'
    content: str

class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str

class ChatResponse(BaseModel):
    model: str
    choices: list[Choice]
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    dropped_messages: int = 0
