from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

@dataclass
class MusicGenGenerationConfiguration:
    DEFAULT_CONFIG_PATH = get_project_directory_path() / "Configurations" / \
        "Music" / "audiocraft" / "MusicGen_generation_configuration.yml"
    
    # See 
    # audiocraft/models/musicgen.py and
    # def set_generation_params(..)
    # top_k used for sampling.
    top_k: Optional[int] = None
    # When set to 0 top_k is used.
    top_p: Optional[float] = None
    # Softmax temperature parameter.
    temperature: Optional[float] = None
    duration: Optional[float] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> \
        'MusicGenGenerationConfiguration':
        path = Path(config_path or cls.DEFAULT_CONFIG_PATH)
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Load and convert values
        for field in fields(cls):
            value = data.get(field.name)
            if value is not None:
                if field.name in ["temperature", "top_p", "duration"]:
                    setattr(data, field.name, float(value))
                elif field.name in ["top_k"]:
                    setattr(data, field.name, int(value))

        return cls(**data)

    def fill_defaults(self) -> 'MusicGenGenerationConfiguration':
        if self.duration is None:
            self.duration = 30.0
        if self.temperature is None:
            self.temperature = 1.0
            
        if self.top_k is None:
            # See def set_generation_params(..) of class MusicGen(BaseGenModel)
            # of musicgen.py in audiocraft.
            self.top_k = 250
            
        if self.top_p is None:
            self.top_p = 0.0
            
        return self

    def get_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                kwargs[field.name] = value
        
        return kwargs
