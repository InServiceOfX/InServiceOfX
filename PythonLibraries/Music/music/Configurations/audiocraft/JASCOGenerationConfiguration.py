from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

@dataclass
class JASCOGenerationConfiguration:
    DEFAULT_CONFIG_PATH = get_project_directory_path() / "Configurations" / \
        "Music" / "audiocraft" / "JASCO_generation_configuration.yml"

    # https://github.com/facebookresearch/audiocraft/blob/main/demos/jasco_demo.ipynb
    # Coefficient used for classifier free guidance - fully conditional term.
    cfg_coef_all: Optional[float] = None
    # Coefficient used for classifier free guidance - additional text
    # conditional term.
    cfg_coef_txt: Optional[float] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> \
        'JASCOGenerationConfiguration':
        path = Path(config_path or cls.DEFAULT_CONFIG_PATH)
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Create a new dictionary for processed values
        processed_data = {}
        
        # Load and convert values
        for field in fields(cls):
            value = data.get(field.name)
            if value is not None:
                if field.name in ["cfg_coef_all", "cfg_coef_txt"]:
                    processed_data[field.name] = float(value)
                else:
                    processed_data[field.name] = value

        return cls(**processed_data)

    def fill_defaults(self) -> 'JASCOGenerationConfiguration':
        if self.cfg_coef_all is None:
            self.cfg_coef_all = 5.0
        if self.cfg_coef_txt is None:
            self.cfg_coef_txt = 0.0
            
        return self

    def get_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                kwargs[field.name] = value
        
        return kwargs
