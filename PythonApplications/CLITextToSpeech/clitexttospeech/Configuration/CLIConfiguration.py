from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import Optional
import yaml

from warnings import warn

class CLIConfiguration(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    text_file_path: Optional[Path] = Field(
        None,
        description="Path to text file with desired speech")

    text_file_paths: Optional[list[Path]] = Field(
        None,
        description="List of paths to text files with desired speech")

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "CLIConfiguration":
        """        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            CLIConfiguration instance. If file doesn't exist or text_file_path
            is empty, returns empty instance.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return cls()
        
        try:
            with file_path.open('r') as f:
                config_data = yaml.safe_load(f) or {}
            
            return cls(**config_data)
                
        except Exception as e:
            # If any error occurs during loading, return empty instance
            warn(
                f"Warning: Could not load configuration from {file_path}: {e}")
            return cls()
    
    def to_yaml(self, file_path: str | Path) -> None:
        """        
        Args:
            file_path: Path where to save the YAML configuration file
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump()

        # Save to YAML
        with file_path.open('w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def update_text_file_path(self, new_path: Optional[Path]) -> None:
        self.text_file_path = new_path
    
    def show_text_file_path(self) -> str:
        """Return a formatted string showing the current text_file_path."""
        if self.text_file_path:
            return f"Current text file path: {self.text_file_path}"
        else:
            return "No text file path set"
