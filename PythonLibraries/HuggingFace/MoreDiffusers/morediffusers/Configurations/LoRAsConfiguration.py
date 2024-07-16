from corecode.FileIO import get_project_directory_path
from pathlib import Path
import re
import yaml

class LoRAsConfiguration:
    def __init__(self, configuration_path):
        self._lora_key_pattern = re.compile(r'lora_(\d+)')
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        data = self._validate_configuration(data)
        self.lora_scale = data["lora_scale"]

        self.lora_scale = self.lora_scale if self.lora_scale is None else \
            float(self.lora_scale)

        self.loras = {}

        for key, lora_parameters in data.items():
            if self._lora_key_pattern.fullmatch(key):
                if "adapter_weight" in lora_parameters.keys():
                    adapter_weight_value = lora_parameters["adapter_weight"]
                    lora_parameters["adapter_weight"] = adapter_weight_value \
                        if adapter_weight_value == None else float(
                            adapter_weight_value)

                self.loras[lora_parameters["adapter_name"]] = lora_parameters

        self.is_to_cuda = data["is_to_cuda"]

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
