from corecode.FileIO import get_project_directory_path
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
import yaml

import torch

@dataclass
class DiffusionPipelineConfiguration:
    # Class variables
    FIELDS_TO_EXCLUDE = ["configuration_path", "cuda_device"]
    DEFAULT_CONFIG_PATH = get_project_directory_path() / "Configurations" / \
        "HuggingFace" / "MoreDiffusers" / "sdxl_pipeline_configuration.yml"
    
    # Instance fields
    configuration_path: Optional[Path] = None
    
    # Model paths and loading
    diffusion_model_path: Optional[Path] = None
    torch_dtype: Optional[torch.dtype] = None
    
    # Hardware optimization
    is_enable_model_cpu_offload: Optional[bool] = None
    is_enable_sequential_cpu_offload: Optional[bool] = None
    is_to_cuda: Optional[bool] = None
    cuda_device: Optional[str] = None

    # Scheduler configuration
    scheduler: Optional[str] = None
    a1111_kdiffusion: Optional[str] = None

    def __post_init__(self):
        self._load_from_yaml(self.configuration_path)
        self.validate_paths()

    def validate_paths(self) -> bool:
        """Validate that required paths exist and are set"""
        if self.diffusion_model_path is None:
            raise ValueError("diffusion_model_path is required but was not set")
        if not self.diffusion_model_path.exists():
            raise RuntimeError(
                f"Path doesn't exist: {self.diffusion_model_path}")
        return True

    def _load_from_yaml(self, config_path: Optional[Path] = None) -> None:
        path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Validate all fields exist in YAML
        for field in fields(self):
            if field.name not in self.FIELDS_TO_EXCLUDE:
                if field.name not in data:
                    raise ValueError(f"{field.name} must be present in configuration file")
        
        # Special validation for required non-None value
        if not data.get("diffusion_model_path"):
            raise ValueError(
                "diffusion_model_path is required and cannot be empty")

        # Load and convert paths
        self.diffusion_model_path = Path(data["diffusion_model_path"])

        # Parse torch dtype
        if data.get("torch_dtype") == "torch.float16":
            self.torch_dtype = torch.float16
        elif data.get("torch_dtype") == "torch.bfloat16":
            self.torch_dtype = torch.bfloat16
        
        # Load boolean flags, only convert explicit true/false values
        for bool_field in ["is_enable_model_cpu_offload", 
                          "is_enable_sequential_cpu_offload", 
                          "is_to_cuda"]:
            value = data.get(bool_field)
            if isinstance(value, str):
                if value.lower() == "true":
                    setattr(self, bool_field, True)
                elif value.lower() == "false":
                    setattr(self, bool_field, False)
            elif isinstance(value, bool):
                setattr(self, bool_field, value)
            else:
                setattr(self, bool_field, None)
        
        # Load scheduler config
        self.scheduler = data.get("scheduler")
        self.a1111_kdiffusion = data.get("a1111_kdiffusion")

        if (getattr(self, "is_to_cuda") is True or 
            getattr(self, "is_enable_model_cpu_offload") is True or
            getattr(self, "is_enable_sequential_cpu_offload") is True):
            # Default to "cuda" if not specified
            self.cuda_device = data.get("cuda_device", "cuda")
        else:
            self.cuda_device = None

    def get_pretrained_kwargs(self) -> dict:
        """Get kwargs dictionary for from_pretrained() method"""
        kwargs = {}
        
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        
        return kwargs

    def get_cuda_device_index(self) -> Optional[int]:
        """Extract device index from cuda_device string.
        Returns None if no index found or cuda_device is None."""
        if not isinstance(self.cuda_device, str):
            return None
        
        if ":" not in self.cuda_device:
            return None
        
        try:
            return int(self.cuda_device.split(":")[1])
        except (IndexError, ValueError):
            return None
