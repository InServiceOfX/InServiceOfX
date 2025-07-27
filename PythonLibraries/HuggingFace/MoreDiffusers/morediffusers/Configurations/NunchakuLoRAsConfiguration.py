from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Optional, List, Tuple, Any, ClassVar, Set
import re
import yaml
from corecode.FileIO import get_project_directory_path

# Move the regex pattern outside the class
LORA_KEY_PATTERN = re.compile(r'lora_(\d+)')

class LoRAParameters(BaseModel):
    """Individual LoRA configuration parameters."""
    # Don't allow extra fields.
    model_config = ConfigDict(extra='forbid')
    
    directory_path: Path = \
        Field(..., description="Directory containing the LoRA file")
    filename: str = Field(..., description="LoRA filename")
    lora_strength: float = \
        Field(..., description="LoRA strength/weight")

    @field_validator('lora_strength', mode='before')
    @classmethod
    def validate_lora_strength(cls, v: Any) -> float:
        """Convert lora_strength to float if it's not already."""
        if v is None:
            raise ValueError("lora_strength cannot be None")
        return float(v)

class NunchakuLoRAsConfiguration(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Instance fields
    lora_scale: Optional[float] = \
        Field(None, description="Global LoRA scale factor")
    loras: Dict[str, LoRAParameters] = \
        Field(
            default_factory=dict,
            description="Dictionary of LoRA configurations")

    # Class variables - annotate with ClassVar
    REQUIRED_LORA_FIELDS: ClassVar[Set[str]] = \
        {"directory_path", "filename", "lora_strength"}

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'NunchakuLoRAsConfiguration':
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")
        
        with config_path.open('r') as f:
            data = yaml.safe_load(f) or {}

        loras = {}
        for key, lora_data in data.items():
            # Use the global pattern instead of class attribute
            if LORA_KEY_PATTERN.fullmatch(key):
                # Validate required fields
                missing_fields = \
                    cls.REQUIRED_LORA_FIELDS - set(lora_data.keys())
                if missing_fields:
                    raise ValueError(
                        f"Missing required fields in '{key}': {missing_fields}")
                
                # Create LoRAParameters object
                lora_params = LoRAParameters(**lora_data)
                loras[lora_params.filename] = lora_params

        if "lora_scale" in data:
            lora_scale = data["lora_scale"]
        else:
            lora_scale = None

        config = cls(loras=loras, lora_scale=lora_scale)

        return config

    def validate_lora_paths(self) -> None:
        """Manually validate that all LoRA files exist.
        Raises ValueError if any LoRA file doesn't exist."""
        errors = []
        
        for _, lora_params in self.loras.items():
            full_path = lora_params.directory_path / lora_params.filename
            if not full_path.exists():
                errors.append(f"LoRA file doesn't exist: {full_path}")
        
        if errors:
            raise ValueError(f"LoRA path validation failed:\n" + "\n".join(errors))

    def get_valid_loras(self) -> List[Tuple[Path, float]]:
        """Returns a list of tuples containing valid LoRA paths and their
        strengths.

        Each tuple contains (full_path: Path, strength: float).
        Only includes LoRAs whose files actually exist on disk.

        Returns:
            List of (Path, float) tuples for valid LoRAs
        """
        valid_loras = []

        for lora_params in self.loras.values():
            full_path = lora_params.directory_path / lora_params.filename
            if full_path.exists():
                valid_loras.append((full_path, lora_params.lora_strength))
        
        return valid_loras

    def get_lora_by_filename(self, filename: str) -> Optional[LoRAParameters]:
        """Get LoRA parameters by filename."""
        return self.loras.get(filename)

    def add_lora(
            self,
            filename: str,
            directory_path: Path,
            lora_strength: float) -> None:
        """Add a new LoRA configuration."""
        lora_params = LoRAParameters(
            directory_path=directory_path,
            filename=filename,
            lora_strength=lora_strength
        )
        self.loras[filename] = lora_params

    def remove_lora(self, filename: str) -> bool:
        """Remove a LoRA configuration by filename.
        
        Returns:
            True if the LoRA was removed, False if it didn't exist
        """
        return self.loras.pop(filename, None) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = self.model_dump()
        
        # Convert loras to the expected YAML format
        loras_dict = {}
        for i, (_, lora_params) in enumerate(self.loras.items()):
            loras_dict[f"lora_{i+1}"] = lora_params.model_dump()
        
        data['loras'] = loras_dict
        return data

    def save_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with config_path.open('w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @staticmethod
    def get_default_config_path() -> Path:
        return get_project_directory_path() / "Configurations" / \
            "HuggingFace" / "MoreDiffusers" / \
            "nunchaku_loras_configuration.yml"
