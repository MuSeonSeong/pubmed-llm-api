"""Public chat contract; stream=true switches the response to server-sent events."""
import time
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, StrictInt, FiniteFloat, model_validator

MAX_IMAGES = 4
CHAT_ID = r'^[A-Za-z0-9_-]{1,64}$'
# How long the model may reason before answering; the token budgets are in app/model_manager.py.
Effort = Literal['none', 'light', 'medium', 'high', 'max']

class LoginRequest(BaseModel):
    name: str = Field(max_length=64)
    password: str = Field(max_length=1024)

class TextPart(BaseModel):
    type: Literal['text']
    text: str

class ImageURL(BaseModel):
    url: str = Field(max_length=12_000_000)
    detail: str | None = None  # sent by OpenAI clients; images are always resized for the model

class ImagePart(BaseModel):
    type: Literal['image_url']
    image_url: ImageURL

ContentPart = Annotated[TextPart | ImagePart, Field(discriminator='type')]

class ChatMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    # Plain text, or OpenAI-style parts so a user message can carry images.
    content: str | list[ContentPart]

    @model_validator(mode='after')
    def check_content(self):
        if isinstance(self.content, str):
            if not self.content:
                raise ValueError('Message content must not be empty')
        elif self.role != 'user':
            raise ValueError('Only user messages may use content parts')
        elif not any(part.type == 'image_url' or part.text.strip() for part in self.content):
            raise ValueError('Message content must not be empty')
        return self

    @property
    def image_count(self) -> int:
        return 0 if isinstance(self.content, str) else sum(part.type == 'image_url' for part in self.content)

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str | None = None  # None: the server's default model (the first in /v1/models)
    messages: list[ChatMessage] = Field(min_length=1)
    # The answer's length; reasoning gets its own budget on top.
    max_tokens: StrictInt = Field(default=512, ge=1, le=4096)
    # None: the model's recommended sampling (0.2 on Transformers models).
    temperature: FiniteFloat | None = Field(default=None, ge=0, le=2, strict=True)
    use_pubmed: bool = True
    stream: bool = False
    # Drop the oldest turns instead of failing when the conversation exceeds the context budget.
    truncate_history: bool = False
    # Let the model reason before answering; the reasoning is returned separately. Ignored by models
    # without reasoning.
    reasoning_effort: Effort | None = None
    # Older clients' switch: true means reasoning_effort medium when reasoning_effort is not given.
    thinking: bool = False
    # Extend the final assistant message instead of starting a new one; only the added text is returned.
    continue_final: bool = False
    # The web UI's chat being answered; while it streams, the user's other devices see it as live.
    chat_id: str | None = Field(default=None, pattern=CHAT_ID)

    @model_validator(mode='after')
    def require_user_query(self):
        if not any(message.role == 'user' for message in self.messages):
            raise ValueError('At least one user message is required')
        if self.continue_final and self.messages[-1].role != 'assistant':
            raise ValueError('continue_final requires the last message to be from the assistant')
        if sum(message.image_count for message in self.messages) > MAX_IMAGES:
            raise ValueError(f'At most {MAX_IMAGES} images per request')
        return self

    @property
    def has_images(self) -> bool:
        return any(message.image_count for message in self.messages)

    @property
    def effort(self) -> str:
        if self.reasoning_effort is not None:
            return self.reasoning_effort
        return 'medium' if self.thinking else 'none'

class ToolTrace(BaseModel):
    name: str
    arguments: dict
    result: dict | None = None
    error: str | None = None

class Source(BaseModel):
    """A PubMed article that a tool actually returned during this answer."""
    pmid: str
    url: str
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    abstract_fetched: bool = False

class AssistantMessage(BaseModel):
    role: Literal['assistant'] = 'assistant'
    content: str
    reasoning_content: str | None = None

class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str

class Usage(BaseModel):
    """Tokens summed over every generation step, including tool-calling rounds."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: 'chatcmpl-' + uuid4().hex)
    object: Literal['chat.completion'] = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    dropped_messages: int = 0
    usage: Usage = Field(default_factory=Usage)
