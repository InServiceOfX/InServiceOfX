from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
from typing import Optional, ClassVar, Dict, Any
import torch
import yaml
from corecode.FileIO import get_project_directory_path

class NunchakuFluxControlConfiguration(BaseModel):
    # Add model_config at class level
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Class variables (using ClassVar to indicate these are class-level)
    DEFAULT_CONFIG_PATH: ClassVar[Path] = get_project_directory_path() / \
        "Configurations" / "HuggingFace" / "MoreDiffusers" / \
        "nunchaku_flux_control_configuration.yml"

    # Instance fields with validation
    flux_model_path: Path = Field(..., description="Path to flux model")
    depth_model_path: Optional[Path] = None
    nunchaku_t5_model_path: Optional[Path] = None
    nunchaku_flux_model_path: Path = Field(
        ...,
        description="Path to nunchaku-flux model")
    torch_dtype: Optional[torch.dtype] = None
    cuda_device: Optional[str] = "cuda"

    # Validators for required paths
    @field_validator('flux_model_path', 'nunchaku_flux_model_path')
    @classmethod
    def validate_model_paths(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path doesn't exist: {v}")
        return v

    @field_validator('torch_dtype', mode='before')
    @classmethod
    def validate_torch_dtype(cls, v: Any) -> Optional[torch.dtype]:
        if isinstance(v, str):
            if v == "torch.float16":
                return torch.float16
            elif v == "torch.bfloat16":
                return torch.bfloat16
            else:
                raise ValueError(f"Unsupported torch_dtype: {v}")
        return v

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> \
        'NunchakuFluxControlConfiguration':
        """Load configuration from YAML file"""
        path = Path(config_path or cls.DEFAULT_CONFIG_PATH)
        
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Only check required fields (non-Optional)
        required_fields = [
            field_name for field_name, field_info in cls.model_fields.items()
            if field_info.is_required()
        ]
        
        missing_fields = \
            [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in configuration: {missing_fields}")
        
        # Convert torch_dtype string to actual torch.dtype if present
        if 'torch_dtype' in data and isinstance(data['torch_dtype'], str):
            if data['torch_dtype'] == "torch.float16":
                data['torch_dtype'] = torch.float16
            elif data['torch_dtype'] == "torch.bfloat16":
                data['torch_dtype'] = torch.bfloat16
            else:
                raise ValueError(
                    f"Unsupported torch_dtype: {data['torch_dtype']}")
        
        return cls(**data)

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