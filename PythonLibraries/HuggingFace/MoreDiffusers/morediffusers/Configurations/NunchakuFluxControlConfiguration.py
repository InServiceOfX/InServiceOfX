from corecode.FileIO import get_project_directory_path
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, ClassVar, Any, Dict
import torch
import yaml

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
        if v is None:
            return None
    
        if isinstance(v, torch.dtype):
            return v
    
        if isinstance(v, str):
            # Remove 'torch.' prefix if present
            dtype_str = v.replace('torch.', '')
            
            dtype_mapping = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32,
                'float64': torch.float64,
                'int4': torch.int4,
                'int8': torch.int8,
                'int16': torch.int16,
                'int32': torch.int32,
                'int64': torch.int64,
                'uint8': torch.uint8,
                'bool': torch.bool,
            }
            
            if dtype_str in dtype_mapping:
                return dtype_mapping[dtype_str]
            else:
                raise ValueError(
                    f"Unsupported torch_dtype: {v}. Supported types: {list(dtype_mapping.keys())}")
        
        raise ValueError(
            f"Invalid torch_dtype: {v}. Must be string or torch.dtype")

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
        
        config = cls(**data)

        return config

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, handling torch.dtype
        conversion."""
        data = self.model_dump()
        
        # Convert torch.dtype back to string for serialization
        if data.get('torch_dtype') is not None:
            data['torch_dtype'] = str(data['torch_dtype'])
        
        return data
