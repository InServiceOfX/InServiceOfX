from corecode.FileIO import get_project_directory_path
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any
import torch
import yaml

class NunchakuConfiguration(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Instance fields with validation
    flux_model_path: Path = \
        Field(..., description="Path to flux model")
    nunchaku_t5_model_path: Optional[Path] = \
        Field(None, description="Path to nunchaku T5 model")
    nunchaku_model_path: Path = \
        Field(..., description="Path to nunchaku model")
    torch_dtype: Optional[torch.dtype] = \
        Field(None, description="Torch data type for model loading")
    cuda_device: Optional[str] = \
        Field("cuda", description="CUDA device specification")

    # Validators
    @field_validator('torch_dtype', mode='before')
    @classmethod
    def validate_torch_dtype(cls, v: Any) -> Optional[torch.dtype]:
        """Convert string torch_dtype to actual torch.dtype object."""
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

    def validate_model_paths(self) -> None:
        """Manually validate that all model paths exist.
        Raises ValueError if any required path doesn't exist."""
        errors = []
        
        # Check required paths
        if not self.flux_model_path.exists():
            errors.append(
                f"flux_model_path doesn't exist: {self.flux_model_path}")
        
        if not self.nunchaku_model_path.exists():
            errors.append(
                f"nunchaku_model_path doesn't exist: {self.nunchaku_model_path}")
        
        # Check optional paths (only if they're not None)
        if self.nunchaku_t5_model_path is not None and \
            not self.nunchaku_t5_model_path.exists():
            errors.append(
                f"nunchaku_t5_model_path doesn't exist: {self.nunchaku_t5_model_path}")
        
        if errors:
            raise ValueError(f"Path validation failed:\n" + "\n".join(errors))

    @classmethod
    def from_yaml(cls, config_path: Path, validate_paths: bool = False) -> 'NunchakuConfiguration':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            validate_paths: If True, validate that all model paths exist after loading
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")
        
        with config_path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Validate required fields
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
        
        # Optionally validate paths
        if validate_paths:
            config.validate_model_paths()
        
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

    def save_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with config_path.open('w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @staticmethod
    def get_default_config_path() -> Path:
        return get_project_directory_path() / "Configurations" / \
            "HuggingFace" / "MoreDiffusers" / "nunchaku_configuration.yml"