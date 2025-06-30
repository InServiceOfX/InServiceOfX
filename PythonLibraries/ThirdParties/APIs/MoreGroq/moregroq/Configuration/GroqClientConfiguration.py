from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union
import yaml

class GroqClientConfiguration(BaseModel):
    """
    Pydantic BaseModel for Groq generation configuration.
    Required fields: model, temperature
    Optional fields: max_tokens, etc.
    """
    
    # Required fields
    model: str = Field(..., description="Groq model to use for generation")
    temperature: float = Field(
        ...,
        description="Sampling temperature, between 0 and 2")
    
    # Optional fields with defaults
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) \
        -> 'GroqClientConfiguration':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            GroqGenerationConfiguration instance
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValidationError: If the YAML data doesn't match the model schema
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                raise ValueError(f"Configuration file {path} is empty")
            
            return cls(**data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {str(e)}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary, excluding None values."""
        config_dict = self.model_dump()
        return {k: v for k, v in config_dict.items() if v is not None}

    def update_chat_completion_configuration(
        self,
        chat_completion_configuration):

        client_configuration_as_dict = self.to_dict()

        for key in client_configuration_as_dict:
            if hasattr(chat_completion_configuration, key) and \
                client_configuration_as_dict[key] is not None:
                setattr(
                    chat_completion_configuration,
                    key,
                    client_configuration_as_dict[key])

        return chat_completion_configuration