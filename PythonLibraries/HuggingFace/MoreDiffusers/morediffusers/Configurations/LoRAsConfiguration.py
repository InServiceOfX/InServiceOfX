from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import re
import yaml
from corecode.FileIO import get_project_directory_path

@dataclass
class LoRAsConfiguration:
    # Class variables
    REQUIRED_LORA_FIELDS = {"directory_path", "weight_name", "adapter_name"}
    
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
        
        # Load lora scale
        if data.get("lora_scale") is not None:
            self.lora_scale = float(data["lora_scale"])
            
        # Load loras
        for key, lora_parameters in data.items():
            if self._lora_key_pattern.fullmatch(key):
                if "adapter_weight" in lora_parameters:
                    value = lora_parameters["adapter_weight"]
                    if value is not None:
                        lora_parameters["adapter_weight"] = float(value)
                self.loras[lora_parameters["adapter_name"]] = lora_parameters

    def _validate_configuration(self, data):
        for key, value in data.items():
            if self._lora_key_pattern.fullmatch(key):
                if 'directory_path' not in value or \
                    'weight_name' not in value or \
                    'adapter_name' not in value:
                    raise ValueError(f"Missing required fields in '{key}'")
        return data

class LoRAsConfigurationForMoreDiffusers(LoRAsConfiguration):
    def __init__(
        self,
        configuration_path=\
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "loras_configuration.yml"
        ):
        super().__init__(configuration_path=configuration_path)
