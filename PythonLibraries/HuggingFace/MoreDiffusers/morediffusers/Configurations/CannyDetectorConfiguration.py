from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationInfo
)

from pathlib import Path
from typing import Optional, ClassVar
from corecode.FileIO import get_project_directory_path
import yaml

class CannyDetectorConfiguration(BaseModel):

    # Class variables
    DEFAULT_CONFIG_PATH: ClassVar[Path] = get_project_directory_path() / \
        "Configurations" / "HuggingFace" / "MoreDiffusers" / \
        "canny_detector_configuration.yml"

    # Required fields with validation and documentation
    low_threshold: int = Field(
        default=100,
        ge=0,
        le=255,
        description="Lower threshold for Canny edge detection (0-255)"
    )
    high_threshold: int = Field(
        default=200,
        ge=0,
        le=255,
        description="Upper threshold for Canny edge detection (0-255)"
    )
    detect_resolution: int = Field(
        default=512,
        gt=0,
        description="Resolution for edge detection"
    )
    image_resolution: int = Field(
        default=512,
        gt=0,
        description="Resolution of output control image"
    )

    # Validators
    @field_validator('high_threshold')
    @classmethod
    def validate_high_threshold(
        cls,
        value: int,
        validation_info: ValidationInfo) -> int:
        """Ensure high_threshold is greater than low_threshold"""
        if 'low_threshold' in validation_info.data and \
            value <= validation_info.data['low_threshold']:
            raise ValueError(
                f"high_threshold ({value}) must be greater than low_threshold ({validation_info.data['low_threshold']})")
        return value

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> 'CannyDetectorConfiguration':
        """Load configuration from YAML file"""
        path = Path(config_path or cls.DEFAULT_CONFIG_PATH)
        
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)

    def to_kwargs(self) -> dict:
        """Convert configuration to dictionary for use as kwargs"""
        return {
            'low_threshold': self.low_threshold,
            'high_threshold': self.high_threshold,
            'detect_resolution': self.detect_resolution,
            'image_resolution': self.image_resolution
        }
