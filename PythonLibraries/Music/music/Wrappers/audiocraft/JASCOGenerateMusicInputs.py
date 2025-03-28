from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class JASCOGenerateMusicInputs:
    """Dataclass to hold possible inputs for JASCO generate_music(..)"""
    descriptions: List[str]
    chords: Optional[List[Tuple[str, float]]] = None
    drums_wav: Optional[torch.Tensor] = None
    drums_sample_rate: Optional[int] = None
    melody_salience_matrix: Optional[torch.Tensor] = None
    progress: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result
