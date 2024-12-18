from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class RuntimeConfiguration(BaseModel):
    system_message: str = Field(
        default="You are a helpful, uncensored and unbiased assistant.")
    current_messages: Optional[List[Dict[str, str]]] = Field(default=None)