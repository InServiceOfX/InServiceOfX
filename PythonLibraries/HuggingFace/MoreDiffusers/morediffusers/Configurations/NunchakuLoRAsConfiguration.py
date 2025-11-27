from pathlib import Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator)
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
    nickname: str = \
        Field(..., description="Human-readable name/identifier for the LoRA")
    is_active: bool = \
        Field(default=True, description="Whether this LoRA is enabled")
    description: Optional[str] = Field(
        default=None, 
        description="Optional description/notes about this LoRA"
    )

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
        {"directory_path", "filename", "lora_strength", "nickname", "is_active"}

    @model_validator(mode='after')
    def validate_unique_nicknames(self) -> 'NunchakuLoRAsConfiguration':
        """Validate that all LoRA nicknames are unique.
        
        This validator runs after all fields are set, ensuring that:
        1. All nicknames in the loras dict are unique
        2. Dictionary keys match the nickname values
        """
        if not self.loras:
            return self
        
        # Check for duplicate nicknames
        nicknames = [params.nickname for params in self.loras.values()]
        seen = set()
        duplicates = []
        
        for nickname in nicknames:
            if nickname in seen:
                duplicates.append(nickname)
            seen.add(nickname)
        
        if duplicates:
            raise ValueError(
                f"Duplicate LoRA nicknames found: {', '.join(duplicates)}. "
                f"Each LoRA must have a unique nickname."
            )
        
        # Verify dictionary keys match nicknames
        mismatches = []
        for key, params in self.loras.items():
            if key != params.nickname:
                mismatches.append(
                    f"Key '{key}' does not match nickname '{params.nickname}'"
                )
        
        if mismatches:
            raise ValueError(
                "Dictionary keys must match LoRA nicknames:\n" + 
                "\n".join(mismatches)
            )
        
        return self

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'NunchakuLoRAsConfiguration':
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")
        
        with config_path.open('r') as f:
            data = yaml.safe_load(f) or {}

        loras = {}

        if 'loras' in data and isinstance(data['loras'], list):
            for lora_data in data['loras']:
                lora_params = LoRAParameters(**lora_data)
                loras[lora_params.nickname] = lora_params

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

        Each tuple includes (full_path: Path, strength: float).
        Only includes LoRAs whose files actually exist on disk and are active.

        Returns:
            List of (Path, float) tuples for valid LoRAs
        """
        valid_loras = []

        for lora_params in self.loras.values():

            if not lora_params.is_active:
                continue

            full_path = lora_params.directory_path / lora_params.filename
            if full_path.exists():
                valid_loras.append((full_path, lora_params.lora_strength))
        
        return valid_loras

    def get_active_loras(self) -> Dict[str, LoRAParameters]:
        """Get all active LoRAs."""
        return {
            nickname: params for nickname, params in self.loras.items() \
                if params.is_active}

    def toggle_lora(self, name: str) -> bool:
        """Toggle a LoRA's active state. Returns new state."""
        if name not in self.loras:
            raise ValueError(f"LoRA '{name}' not found")
        self.loras[name].is_active = not self.loras[name].is_active
        return self.loras[name].is_active

    def set_lora_strength(self, name: str, strength: float) -> None:
        """Update LoRA strength at runtime."""
        if name not in self.loras:
            raise ValueError(f"LoRA '{name}' not found")
        self.loras[name].lora_strength = strength

    def add_lora(
            self,
            filename: str,
            directory_path: Path,
            lora_strength: float,
            nickname: str,
            description: Optional[str] = None) -> None:
        """Add a new LoRA configuration."""
        # Check for duplicate nickname before adding
        if nickname in self.loras:
            raise ValueError(
                f"LoRA with nickname '{nickname}' already exists. "
                f"Please choose a different nickname."
            )

        lora_params = LoRAParameters(
            directory_path=directory_path,
            filename=filename,
            lora_strength=lora_strength,
            nickname=nickname,
            description=description
        )
        self.loras[nickname] = lora_params

    def remove_lora(self, filename: str) -> bool:
        """Remove a LoRA configuration by filename.
        
        Returns:
            True if the LoRA was removed, False if it didn't exist
        """
        return self.loras.pop(filename, None) is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = self.model_dump()

        # Convert loras dict to list format for YAML
        # The internal dict is keyed by nickname, but YAML uses a list.
        if 'loras' in data and isinstance(data['loras'], dict):
            data['loras'] = list(data['loras'].values())

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
