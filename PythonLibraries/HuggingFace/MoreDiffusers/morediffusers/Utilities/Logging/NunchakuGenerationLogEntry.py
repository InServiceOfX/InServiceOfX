from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

from .LoRALogEntry import LoRALogEntry

class NunchakuGenerationLogEntry(BaseModel):
    """Single Nunchaku generation log entry."""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Nunchaku model info
    nunchaku_model_parent_dir: str
    nunchaku_model_filename: Optional[str] = None

    # Flux generation parameters
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    guidance_scale_used: Optional[float] = None

    # LoRAs used
    loras: Optional[List[LoRALogEntry]] = None

    # Prompts
    prompt: str
    prompt_2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None

    generation_hash: Optional[str] = None
    truncated_generation_hash: Optional[str] = None