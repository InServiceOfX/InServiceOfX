from pathlib import Path
import json
from typing import Dict, List, Optional, Union

class JSONFile:
    """Handles file input/output operations for a JSON file."""
    
    @staticmethod
    def ensure_directory_exists(directory_path: Path) -> None:
        """Ensure that a directory exists, creating it if necessary."""
        directory_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_json(file_path: Path) -> Optional[Dict]:
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Empty file
                    return None
                return json.loads(content)
        except json.JSONDecodeError:
            # Invalid JSON
            return None
    
    @staticmethod
    def save_json(file_path: Path, data: Union[Dict, List]) -> bool:
        try:
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
