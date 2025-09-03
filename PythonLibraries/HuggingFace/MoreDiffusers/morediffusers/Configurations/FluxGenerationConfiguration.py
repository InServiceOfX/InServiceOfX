from corecode.FileIO import get_project_directory_path
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, ClassVar, Set
import yaml

class FluxGenerationConfiguration(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Class variables
    FIELDS_TO_EXCLUDE: ClassVar[Set[str]] = {"temporary_save_path", "seed"}

    # Generation parameters
    # See pipeline_flux.py, class FluxPipeline(..), def __call__(..),

    # True classifier-free guidance (guidance_scale) is enabled when
    # true_cfg_scale > 1 and negative_prompt is provided.
    true_cfg_scale: Optional[float] = \
        Field(None, description="True classifer-free guidance scale")
    height: Optional[int] = Field(None, description="Image height")
    width: Optional[int] = Field(None, description="Image width")
    num_inference_steps: Optional[int] = \
        Field(None, description="Number of inference steps")
    num_images_per_prompt: Optional[int] = \
        Field(None, description="Number of images per prompt")
    seed: Optional[int] = Field(None, description="Random seed")
    # pipeline_flux.py, class FluxPipeline(..), docstring: Higher 'guidance
    # scale' encourages model to generate images more aligned with 'prompt' at
    # expense of lower image quality.
    guidance_scale: Optional[float] = \
        Field(None, description="Guidance scale for generation")
    # Default value is 512, which is the maximum that can be used without a
    # runtime error.
    max_sequence_length: Optional[int] = \
        Field(None, description="Maximum sequence length")
    temporary_save_path: Path = \
        Field(default_factory=Path.cwd, description="Temporary save path")

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) \
        -> 'FluxGenerationConfiguration':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        path = Path(config_path or cls.get_default_config_path())
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with path.open('r') as f:
            data = yaml.safe_load(f) or {}
        
        # Validate all fields except excluded ones exist in YAML
        for field_name in cls.model_fields.keys():
            if field_name not in cls.FIELDS_TO_EXCLUDE:
                if field_name not in data:
                    raise ValueError(
                        f"{field_name} must be present in configuration file")
        
        # Load and convert numeric values
        processed_data = {}
        for field_name in cls.model_fields.keys():
            if field_name not in cls.FIELDS_TO_EXCLUDE:
                value = data.get(field_name)
                if value is not None:
                    if field_name == "guidance_scale":
                        processed_data[field_name] = float(value)
                    else:
                        processed_data[field_name] = int(value)
                else:
                    processed_data[field_name] = None

        processed_data["temporary_save_path"] = Path(
            data.get("temporary_save_path", Path.cwd()))

        if "seed" in data:
            processed_data["seed"] = int(data["seed"])

        return cls(**processed_data)

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get kwargs dictionary for pipeline __call__() method"""
        kwargs = {}
        
        for field_name in self.model_fields.keys():
            if field_name not in self.FIELDS_TO_EXCLUDE:
                value = getattr(self, field_name)
                if value is not None:
                    kwargs[field_name] = value
        
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    @staticmethod
    def get_default_config_path() -> Path:
        return get_project_directory_path() / "Configurations" / \
            "HuggingFace" / "MoreDiffusers" / \
                "flux_generation_configuration.yml"
