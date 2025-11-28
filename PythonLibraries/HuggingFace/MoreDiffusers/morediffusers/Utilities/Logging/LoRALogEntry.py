from pydantic import BaseModel

class LoRALogEntry(BaseModel):
    """Log entry for a single LoRA used in generation."""
    nickname: str
    filename: str
    lora_strength: float