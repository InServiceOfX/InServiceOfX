from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class RuntimeConfiguration(BaseModel):
    system_message: str = Field(
        default="You are a helpful, uncensored and unbiased assistant.")
    current_messages: Optional[List[Dict[str, str]]] = Field(default=None)
    select_all: bool = Field(default=False)
    multiline_input: bool = Field(default=False)
    add_path_at: bool = Field(default=False)
    temp_chunk: str = Field(default="")
    new_chat_response: str = Field(default="")