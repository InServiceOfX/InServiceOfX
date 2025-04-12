from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, ClassVar, Union
import torch
import yaml

@dataclass
class Configuration:
    """Configuration class for model settings."""
    
    # Class constants
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path(
        get_project_directory_path() / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "configuration.yml"
    )
    
    # Instance fields with defaults for empty construction
    configuration_path: Optional[Path] = None
    task: Optional[str] = None
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    torch_dtype: Union[str, torch.dtype] = "torch.float16"
    device_map: Optional[str] = "auto"
    
    def __post_init__(self):
        """Initialize after construction."""
        # If configuration_path is provided, load from YAML
        if self.configuration_path is not None:
            self._load_from_yaml()
        
        # Process model_name if model_path is set but model_name isn't
        if self.model_path and not self.model_name:
            self.model_name = Path(self.model_path).name
            
        # Convert torch_dtype string to actual torch dtype
        self._process_torch_dtype()
    
    @classmethod
    def from_yaml(
        cls,
        configuration_path: Optional[Path] = None) -> 'Configuration':
        """Create a Configuration instance from a YAML file."""
        path = configuration_path or cls.DEFAULT_CONFIG_PATH
        return cls(configuration_path=path)
    
    def _load_from_yaml(self) -> None:
        """Load configuration from YAML file."""
        path = self.configuration_path or self.DEFAULT_CONFIG_PATH
        
        try:
            with open(str(path), 'r') as f:
                data = yaml.safe_load(f)
                
            # Update instance attributes from YAML data
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        except yaml.YAMLError:
            raise ValueError(f"Invalid YAML in configuration file: {path}")
    
    def _process_torch_dtype(self) -> None:
        """Convert torch_dtype string to actual torch dtype."""
        if isinstance(self.torch_dtype, str):
            if self.torch_dtype == "torch.float16":
                self.torch_dtype = torch.float16
            elif self.torch_dtype == "torch.bfloat16":
                self.torch_dtype = torch.bfloat16
            elif self.torch_dtype == "torch.float32":
                self.torch_dtype = torch.float32
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        # Use __dict__ but filter out None values and private attributes
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and v is not None}
