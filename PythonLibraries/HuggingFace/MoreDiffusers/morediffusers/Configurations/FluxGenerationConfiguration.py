from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class FluxGenerationConfiguration:
    # Class variables
    FIELDS_TO_EXCLUDE = ["configuration_path", "temporary_save_path"]
    DEFAULT_CONFIG_PATH = get_project_directory_path() / "Configurations" / \
        "HuggingFace" / "MoreDiffusers" / "flux_generation_configuration.yml"
    
    # Instance fields
    configuration_path: Optional[Path] = None
    
    # Generation parameters
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    num_images_per_prompt: Optional[int] = None
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    # Default value is 512, which is the maximum that can be used without a
    # runtime error.
    max_sequence_length: Optional[int] = None
    temporary_save_path: Path = field(default_factory=Path.cwd)

    def __post_init__(self):
        self._load_from_yaml(self.configuration_path)

    def _load_from_yaml(self, config_path: Optional[Path] = None) -> None:
        path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Validate all fields except excluded ones exist in YAML
        for field in fields(self):
            if field.name not in self.FIELDS_TO_EXCLUDE or \
                field.name == "temporary_save_path":
                if field.name not in data:
                    raise ValueError(
                        f"{field.name} must be present in configuration file")
        
        # Load and convert numeric values
        for field in fields(self):
            if field.name not in self.FIELDS_TO_EXCLUDE or \
                field.name == "temporary_save_path":
                value = data.get(field.name)
                if value is not None:
                    if field.name == "guidance_scale":
                        setattr(self, field.name, float(value))
                    elif field.name == "temporary_save_path":
                        setattr(self, field.name, Path(value))
                    else:
                        setattr(self, field.name, int(value))

    def get_generation_kwargs(self) -> dict:
        """Get kwargs dictionary for pipeline __call__() method"""
        kwargs = {}
        
        for field in fields(self):
            if (field.name not in self.FIELDS_TO_EXCLUDE and \
                field.name != "seed"):
                value = getattr(self, field.name)
                if value is not None:
                    kwargs[field.name] = value
        
        return kwargs
