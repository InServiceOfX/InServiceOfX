from clichat.Configuration.CLIChatConfiguration import CLIChatConfiguration
from clichat.Utilities import get_environment_variable
from pathlib import Path
from typing import Optional

import yaml
import os

class Configuration:
    @staticmethod
    def _get_default_configuration_path() -> Path:
        return Path.home() / ".config" / "clichat" / "Configurations" / \
            "clichat_configuration.yml"

    @staticmethod
    def load_yaml(configuration_path: Optional[Path] = None) -> dict:
        if configuration_path is None:
            configuration_path = Configuration._get_default_configuration_path()
        
        try:
            with open(str(configuration_path), 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"\nError parsing configuration file: {configuration_path}")
            print("Make sure the YAML syntax is correct.")
            print(f"Details: {str(e)}\n")
            raise SystemExit(1)
        except FileNotFoundError:
            print(f"\nConfiguration file not found: {configuration_path}")
            print("Make sure the file exists and you have read permissions.\n")
            raise SystemExit(1)

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self._get_default_configuration_path()
        
        self.configuration_path = config_path
        yaml_data = self.load_yaml(config_path)
        configuration = CLIChatConfiguration(**yaml_data)
        
        # Transfer all attributes from pydantic model to this instance
        for key, value in configuration.model_dump().items():
            setattr(self, key, value)