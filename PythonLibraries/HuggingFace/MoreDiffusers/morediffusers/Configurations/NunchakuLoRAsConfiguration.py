from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import re
import yaml
from corecode.FileIO import get_project_directory_path

@dataclass
class NunchakuLoRAsConfiguration:
    # Class variables
    REQUIRED_LORA_FIELDS = {"directory_path", "filename", "lora_strength"}
    
    # Instance fields
    configuration_path: Path
    lora_scale: Optional[float] = None
    loras: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        self._lora_key_pattern = re.compile(r'lora_(\d+)')
        self._load_from_yaml()
    
    def _load_from_yaml(self) -> None:
        with self.configuration_path.open('r') as f:
            data = yaml.safe_load(f) or {}
            
        data = self._validate_configuration(data)
            
        # Load loras
        for key, lora_parameters in data.items():
            if self._lora_key_pattern.fullmatch(key):
                if "lora_strength" in lora_parameters:
                    value = lora_parameters["lora_strength"]
                    if value is not None:
                        lora_parameters["lora_strength"] = float(value)
                self.loras[lora_parameters["filename"]] = lora_parameters

    def _validate_configuration(self, data):
        for key, value in data.items():
            if self._lora_key_pattern.fullmatch(key):
                if 'directory_path' not in value or \
                    'filename' not in value or \
                    'lora_strength' not in value:
                    raise ValueError(f"Missing required fields in '{key}'")
        return data

    def get_valid_loras(self) -> list[tuple[Path, float]]:
        """Returns a list of tuples containing valid LoRA paths and their
        strengths.

        Each tuple contains (full_path: Path, strength: float).
        Only includes LoRAs whose files actually exist on disk.

        Returns:
            List of (Path, float) tuples for valid LoRAs
        """
        valid_loras = []

        for lora_data in self.loras.values():
            full_path = Path(lora_data["directory_path"]) / \
                lora_data["filename"]
            if full_path.exists():
                valid_loras.append(
                    (full_path, float(lora_data["lora_strength"]))
                )
            else:
                print(f"LoRA file not found: {full_path}")
        return valid_loras

class NunchakuLoRAsConfigurationForMoreDiffusers(NunchakuLoRAsConfiguration):
    def __init__(
        self,
        configuration_path=\
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "nunchaku_loras_configuration.yml"
        ):
        super().__init__(configuration_path=configuration_path)
