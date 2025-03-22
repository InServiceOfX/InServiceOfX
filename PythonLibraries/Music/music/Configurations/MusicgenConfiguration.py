from dataclasses import dataclass
from pathlib import Path
import torch
import yaml
from typing import Optional, Dict, Any, Union

@dataclass
class MusicgenConfiguration:
    pretrained_model_name_or_path: str
    device_map: Optional[str] = None
    attn_implementation: Optional[str] = None
    torch_dtype: Optional[Any] = None
    
    def __post_init__(self):
        path = Path(self.pretrained_model_name_or_path)
        if not path.is_absolute():
            path = Path.cwd() / path
                
        self.pretrained_model_name_or_path = str(path)
    
    def fill_defaults(self) -> 'MusicgenConfiguration':
        if self.attn_implementation is None:
            self.attn_implementation = "eager"
        
        if self.torch_dtype is None:
            self.torch_dtype = torch.float32
            
        return self
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        
        # Include optional parameters only if they have values
        if self.device_map is not None and self.device_map != "":
            kwargs["device_map"] = self.device_map
            
        if self.attn_implementation is not None and \
            self.attn_implementation != "":
            kwargs["attn_implementation"] = self.attn_implementation
            
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
            
        return kwargs
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MusicgenConfiguration':
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"YAML configuration file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert string representation of torch_dtype to actual torch dtype if needed
        if isinstance(config_dict.get("torch_dtype"), str):
            dtype_str = config_dict["torch_dtype"]
            if dtype_str == "float16" or \
                dtype_str == "half" or \
                dtype_str == "torch.float16":
                config_dict["torch_dtype"] = torch.float16
            elif dtype_str == "float32" or \
                dtype_str == "float" or \
                dtype_str == "torch.float32":
                config_dict["torch_dtype"] = torch.float32
            elif dtype_str == "bfloat16" or \
                dtype_str == "torch.bfloat16":
                config_dict["torch_dtype"] = torch.bfloat16
            elif dtype_str == "none" or dtype_str == "":
                config_dict["torch_dtype"] = None

        return cls(**config_dict)
