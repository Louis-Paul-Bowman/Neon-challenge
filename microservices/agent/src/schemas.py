from pydantic import BaseModel, Field


class Scrambled(BaseModel):
    word: str
    timestamp: int


class PromptRequest(BaseModel):
    type: str | None = None
    prompt: str | list[Scrambled]
    thread_id: str | None = Field(
        default=None, description="UUID4 session ID for recall across turns"
    )
