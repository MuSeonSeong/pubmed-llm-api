"""Public chat contract; stream=true switches the response to server-sent events."""
import time
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, StrictInt, FiniteFloat, model_validator

MAX_IMAGES = 4

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
    model: str = 'qwen3.5-2b'
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: StrictInt = Field(default=512, ge=1, le=2048)
    temperature: FiniteFloat = Field(default=0.2, ge=0, le=2, strict=True)
    use_pubmed: bool = True
    stream: bool = False
    # Drop the oldest turns instead of failing when the conversation exceeds the context budget.
    truncate_history: bool = False
    # Let the model reason before answering; the reasoning is returned separately.
    thinking: bool = False
    # Extend the final assistant message instead of starting a new one; only the added text is returned.
    continue_final: bool = False

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
