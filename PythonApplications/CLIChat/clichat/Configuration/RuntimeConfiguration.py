from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class RuntimeConfiguration(BaseModel):
    current_messages: Optional[List[Dict[str, str]]] = Field(default=None)
    multiline_input: bool = Field(default=False)
    add_path_at: bool = Field(default=False)
    temp_chunk: str = Field(default="")
    new_chat_response: str = Field(default="")
    wrap_words: bool = Field(default=True)
    model: str = Field(default="llama-3.3-70b-versatile")
    # https://console.groq.com/docs/api-reference#chat-create
    # max_tokens integer or null Optional
    # Max number of tokens that can be generated in chat completion. Total
    # length of input tokens and generated tokens is limited by model's context
    # length.
    max_tokens: int | None = Field(default=None)
