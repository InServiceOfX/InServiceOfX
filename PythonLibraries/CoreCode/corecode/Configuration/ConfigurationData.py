from dataclasses import dataclass, field
from typing import Dict, Optional, Any

@dataclass
class ConfigurationData:
    """Internal representation of the project's .config file."""
    BASE_DATA_PATH: str = ""

    PROMPTS_COLLECTION_PATH: Optional[str] = None

    numbered_data_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert the configuration data to a dictionary."""
        return {
            "BASE_DATA_PATH": self.BASE_DATA_PATH,
            "PROMPTS_COLLECTION_PATH": self.PROMPTS_COLLECTION_PATH,
            **self.numbered_data_paths
        }

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        
        # Handle BASE_DATA_PATH (required, can be empty)
        if self.BASE_DATA_PATH:
            result['BASE_DATA_PATH'] = self.BASE_DATA_PATH
        else:
            result['BASE_DATA_PATH'] = ""
        
        # Handle all numbered BASE_DATA_PATH fields dynamically
        for key, value in self.numbered_data_paths.items():
            if value is not None and value.strip():
                result[key] = value
        
        if self.PROMPTS_COLLECTION_PATH is not None or not "":
            result['PROMPTS_COLLECTION_PATH'] = self.PROMPTS_COLLECTION_PATH
        
        return result

    def add_numbered_data_path(self, numbered_key: str, value: str) -> None:
        """Add a numbered data path (e.g., BASE_DATA_PATH_1,
        BASE_DATA_PATH_42)."""
        self.numbered_data_paths[numbered_key] = \
            value if value.strip() else None