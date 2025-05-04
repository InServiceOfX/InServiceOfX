from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Dict, Any
from pathlib import Path
import yaml

class CLIConfiguration(BaseModel):
    # Command settings
    exit_command: str = Field(default=".exit")
    help_command: str = Field(default=".help")
    
    # UI settings
    user_color: str = Field(default="ansigreen")
    assistant_color: str = Field(default="ansiblue")
    system_color: str = Field(default="ansiyellow")
    info_color: str = Field(default="ansicyan")
    error_color: str = Field(default="ansired")

    file_history_path: Optional[Path] = None

    terminal_CommandEntryColor2: str = Field(default="ansigreen")
    terminal_PromptIndicatorColor2: str = Field(default="ansicyan")

    inference_mode: str = Field(default="sglang")

    def __init__(self, is_dev: bool = False, **data):
        super().__init__(**data)

        if "file_history_path" not in data or data["file_history_path"] is None:
            if is_dev:
                self.file_history_path = Path(__file__).parents[2] / \
                    "Configurations" / "file_history.txt"
            else:
                self.file_history_path = Path.home() / ".clichatlocal" / \
                    "file_history.txt"
    
    @classmethod
    def from_yaml(cls, file_path: Path, is_dev: bool = False) -> "CLIConfiguration":
        """
        Args:
            file_path: Path to the YAML configuration file
        """
        try:
            # Check if file exists
            if not file_path.exists():
                print(
                    f"Warning: Configuration file {file_path} not found. Using default values.")
                return cls(is_dev=is_dev)
            
            # Load YAML file
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Handle None case
            if config_data is None:
                print(
                    f"Warning: Configuration file {file_path} is empty. Using default values.")
                return cls(is_dev=is_dev)
            
            # Create configuration with loaded values
            config = cls(is_dev=is_dev, **config_data)
            
            return config
            
        except Exception as e:
            print(f"Error loading configuration from {file_path}: {str(e)}")
            print("Using default configuration values.")
            return cls(is_dev=is_dev)

    