from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import warnings

class JSONFile:
    """Handles file input/output operations for a JSON file."""

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
        except Exception as e:
            warnings.warn(f"Error saving JSON file: {e}")
            return False
