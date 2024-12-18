from corecode.FileIO import get_project_directory_path
from clichat.Configuration.CLIChatConfiguration import CLIChatConfiguration
from pathlib import Path
from typing import Optional

import yaml

class Configuration:
    @staticmethod
    def load_yaml(configuration_path: Optional[Path] = None) -> dict:
        if configuration_path is None:
            configuration_path = (
                get_project_directory_path() 
                    / "PythonApplications" 
                    / "CLIChat" 
                    / "Configurations" 
                    / "clichat_configuration.yml")
        
        with open(str(configuration_path), 'r') as f:
            return yaml.safe_load(f)
    
    def __init__(self, configuration_path: Optional[Path] = None):
        yaml_data = self.load_yaml(configuration_path)
        configuration = CLIChatConfiguration(**yaml_data)
        
        # Transfer all attributes from pydantic model to this instance
        for key, value in configuration.model_dump().items():
            setattr(self, key, value)