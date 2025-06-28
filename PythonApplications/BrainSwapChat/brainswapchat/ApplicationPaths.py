from pathlib import Path
from dataclasses import dataclass, fields
from typing import Dict, Optional

@dataclass
class ApplicationPaths:
    system_messages_file_path: Path

    @classmethod
    def create_path_names(
        cls,
        new_project_root_path: Optional[str | Path] = None,
        is_development: bool = False) \
            -> 'ApplicationPaths':

        app_path = Path(__file__).resolve().parents[1]

        if is_development:
            system_messages_file_path = \
                app_path / "Configurations" / "system_messages.json"
        elif new_project_root_path is not None:
            system_messages_file_path = \
                new_project_root_path / "Configurations" / \
                    "system_messages.json"
        else:
            system_messages_file_path = \
                Path.home() / ".config" / "brainswapchat" / \
                    "system_messages.json"

        return cls(system_messages_file_path=system_messages_file_path)

    def check_paths_exist(self) -> Dict[str, bool]:
        """
        Check if each data member path exists as a file.
        
        Returns:
            Dictionary with data member names as keys and boolean existence as
            values.
        """
        result = {}
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            
            if isinstance(field_value, Path):
                result[field_name] = field_value.exists()
            else:
                # If it's not a Path, assume it doesn't exist
                result[field_name] = False
                
        return result
    
    def create_missing_files(self, *path_names: str) -> Dict[str, bool]:
        """
        Create files for specified data member paths if they don't exist.
        
        Args:
            *path_names: Variable number of data member names to create
            
        Returns:
            Dictionary with data member names as keys and creation success as values.
        """
        result = {}
        
        for path_name in path_names:
            try:
                # Check if the attribute exists
                if not hasattr(self, path_name):
                    result[path_name] = False
                    continue
                
                field_value = getattr(self, path_name)
                
                if not isinstance(field_value, Path):
                    result[path_name] = False
                    continue
                
                # Create parent directories if they don't exist
                field_value.parent.mkdir(parents=True, exist_ok=True)
                
                # Create the file if it doesn't exist.
                if not field_value.exists():
                    field_value.touch()
                    result[path_name] = True
                else:
                    # File already exists
                    result[path_name] = True
                    
            except Exception as e:
                print(f"Error creating path {path_name}: {e}")
                result[path_name] = False
                
        return result
    
    def create_all_missing_paths(self) -> Dict[str, bool]:
        """
        Returns:
            Dictionary with all data member names as keys and creation success
            as values.
        """
        all_path_names = [field.name for field in fields(self)]
        return self.create_missing_paths(*all_path_names) 