from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Union, Literal
import yaml

class TextToSpeechConfiguration(BaseModel):
    """
    Configuration for xAI Text-to-Speech API.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Text input settings (optional - can also pass text directly to client)
    text_file_path: Path = Field(
        default=None,
        description="Path to text file with desired speech"
    )
    
    text_file_paths: Optional[list[Path]] = Field(
        default=None,
        description="List of paths to text files with desired speech"
    )

    # API settings
    base_url: Optional[str] = Field(
        default="https://api.x.ai/v1",
        description="Base URL for xAI API"
    )
    
    # Voice settings
    voice: Literal["Ara", "Rex", "Sal", "Eve", "Una", "Leo"] = Field(
        default="Ara",
        description="Voice to use for speech synthesis"
    )
    
    # Audio format
    response_format: Literal["mp3", "wav", "opus", "flac", "pcm"] = Field(
        default="mp3",
        description="Audio format for output"
    )

    # Output settings
    output_directory: Optional[Path] = Field(
        default=None,
        description=(
            "Directory to save audio files. If None, uses current directory.")
    )
    
    filename_prefix: Optional[str] = Field(
        default="speech",
        description="Prefix for auto-generated filenames"
    )
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) \
        -> 'XAITextToSpeechConfiguration':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            XAITextToSpeechConfiguration instance
            
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
            
            # Convert string paths to Path objects
            if 'output_directory' in data and data['output_directory']:
                data['output_directory'] = Path(data['output_directory'])

            if 'text_file_path' in data and data['text_file_path']:
                data['text_file_path'] = Path(data['text_file_path'])
            
            if 'text_file_paths' in data and data['text_file_paths']:
                data['text_file_paths'] = \
                    [Path(p) for p in data['text_file_paths']]

            return cls(**data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {str(e)}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary, excluding None values."""
        config_dict = self.model_dump()
        # Convert Path objects back to strings for serialization
        if config_dict.get('output_directory'):
            config_dict['output_directory'] = str(
                config_dict['output_directory'])
        return {k: v for k, v in config_dict.items() if v is not None}