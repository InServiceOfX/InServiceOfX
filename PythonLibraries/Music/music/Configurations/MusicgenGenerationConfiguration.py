from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

@dataclass
class MusicgenGenerationConfiguration:
    DEFAULT_CONFIG_PATH = get_project_directory_path() / "Configurations" / \
        "Music" / "musicgen_generation_configuration.yml"
    
    # Generation parameters
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> \
        'MusicgenGenerationConfiguration':
        path = Path(config_path or cls.DEFAULT_CONFIG_PATH)
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Load and convert values
        for field in fields(cls):
            value = data.get(field.name)
            if value is not None:
                if field.name in ["temperature", "top_p"]:
                    setattr(data, field.name, float(value))
                elif field.name in ["max_new_tokens", "top_k"]:
                    setattr(data, field.name, int(value))

        return cls(**data)

    def fill_defaults(self) -> 'MusicgenGenerationConfiguration':
        if self.max_new_tokens is None:
            self.max_new_tokens = 512
        
        if self.temperature is None:
            self.temperature = 1.0
            
        if self.top_k is None:
            self.top_k = 50
            
        if self.top_p is None:
            self.top_p = 1.0
            
        return self

    def get_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                kwargs[field.name] = value
        
        return kwargs
