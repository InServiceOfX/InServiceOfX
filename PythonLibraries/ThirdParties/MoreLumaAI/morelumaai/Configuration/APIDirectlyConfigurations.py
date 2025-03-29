from typing import Optional, Dict
import requests
import time
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Ray2Configuration:
    """Configuration for Ray 2 Image to Video generation"""
    model: str = "ray-2"
    aspect_ratio: Optional[str] = None
    fps: Optional[int] = None
    duration: Optional[float] = None
    loop: Optional[bool] = None
    
    def __post_init__(self):
        """Validate the model is a supported Ray model"""
        supported_models = ["ray-2", "ray-2-flash"]
        if self.model not in supported_models:
            raise ValueError(
                f"Model must be one of {supported_models}, got {self.model}")
    
    def to_api_kwargs(self) -> Dict:
        """Convert configuration to API parameters"""
        kwargs = {"model": self.model}
        if self.aspect_ratio:
            kwargs["aspect_ratio"] = self.aspect_ratio
        if self.fps:
            kwargs["fps"] = self.fps
        if self.duration:
            kwargs["duration"] = self.duration
        if self.loop is not None:
            kwargs["loop"] = self.loop
        return kwargs

