from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Dict, Union
from warnings import warn
import yaml

class ModelList(BaseModel):
    """Configuration class for managing model names and their file paths"""
    
    models: Dict[str, Path] = Field(
        default_factory=dict,
        description="Dictionary mapping model names to their file paths"
    )
    
    @validator('models', pre=True)
    def convert_strings_to_paths(cls, v):
        """Convert string paths to Path objects and validate they exist"""
        if isinstance(v, dict):
            converted = {}
            for model_name, path_str in v.items():
                if isinstance(path_str, str):
                    path = Path(path_str)
                    converted[model_name] = path
                elif isinstance(path_str, Path):
                    converted[model_name] = path_str
                else:
                    raise ValueError(
                        f"Invalid path type for model '{model_name}': {type(path_str)}")
            return converted
        return v
    
    @validator('models')
    def validate_paths_exist(cls, v):
        """Validate that all model paths exist"""
        for model_name, path in v.items():
            if not path.exists():
                raise ValueError(
                    f"Model path does not exist for '{model_name}': {path}")
        return v
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "ModelList":
        """
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If any model paths don't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                warn(f"Warning: Configuration file {file_path} is empty. Using default values.")
                return cls()
            
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {file_path}: {str(e)}")
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Args:
            file_path: Path where to save the YAML file
        """
        file_path = Path(file_path)
        
        # Convert Path objects back to strings for YAML serialization
        config_data = {
            'models': {
                name: str(path) for name, path in self.models.items()
            }
        }
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ValueError(
                f"Error saving configuration to {file_path}: {str(e)}")
    
    def add_model(self, name: str, path: Union[str, Path]) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValueError(f"Model path does not exist: {path_obj}")
        
        self.models[name] = path_obj
    
    def remove_model(self, name: str) -> None:
        if name in self.models:
            del self.models[name]
        else:
            raise KeyError(f"Model '{name}' not found in configuration")
    
    def get_model_path(self, name: str) -> Path:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in configuration")
        return self.models[name]
    
    def list_models(self) -> list[str]:
        return list(self.models.keys())
    
    def __len__(self) -> int:
        return len(self.models)
    
    def __contains__(self, name: str) -> bool:
        return name in self.models
