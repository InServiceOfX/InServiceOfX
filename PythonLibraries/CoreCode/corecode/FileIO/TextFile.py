from pathlib import Path
from typing import List, Optional
import warnings

class TextFile:
    """Handles file input/output operations for text files."""

    @staticmethod
    def load_text(file_path: Path) -> Optional[str]:
        """
        Load text content from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            str: File content as string, or None if file doesn't exist or is empty
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:  # Empty file
                    return None
                return content
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read().strip()
                    if not content:
                        return None
                    return content
            except Exception as e:
                warnings.warn(
                    f"Error reading text file with latin-1 encoding: {e}")
                return None
        except Exception as e:
            warnings.warn(f"Error reading text file: {e}")
            return None
    
    @staticmethod
    def save_text(file_path: Path, content: str) -> bool:
        """
        Args:
            file_path: Path where to save the text file
            content: String content to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            warnings.warn(f"Error saving text file: {e}")
            return False
    
    @staticmethod
    def load_lines(file_path: Path) -> Optional[List[str]]:
        """
        Load text file as a list of lines.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List[str]: List of lines, or None if file doesn't exist
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Remove trailing whitespace and empty lines at the end
                while lines and lines[-1].strip() == '':
                    lines.pop()
                return lines
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    while lines and lines[-1].strip() == '':
                        lines.pop()
                    return lines
            except Exception as e:
                warnings.warn(f"Error reading text file with latin-1 encoding: {e}")
                return None
        except Exception as e:
            warnings.warn(f"Error reading text file: {e}")
            return None
    
    @staticmethod
    def save_lines(file_path: Path, lines: List[str]) -> bool:
        """
        Save a list of strings as lines in a text file.
        
        Args:
            file_path: Path where to save the text file
            lines: List of strings to save as lines
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line)
                    if not line.endswith('\n'):
                        f.write('\n')
            return True
        except Exception as e:
            warnings.warn(f"Error saving text file: {e}")
            return False
