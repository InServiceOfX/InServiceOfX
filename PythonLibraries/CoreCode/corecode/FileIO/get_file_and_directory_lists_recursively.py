from pathlib import Path
from typing import List, Optional, Union
import fnmatch
import os

def get_file_and_directory_lists_recursively(
    path: Optional[Union[str, Path]] = None,
    gitignore_path: Optional[Union[str, Path]] = None
) -> List[Path]:
    """
    Recursively list all files using modern pathlib approach.
    
    Args:
        path: Directory to search (defaults to current working directory)
        gitignore_path: Path to .gitignore file (optional)
        
    Returns:
        List of Path objects representing all files found
    """
    # Convert path to Path object, default to current directory
    if path is None:
        search_path = Path.cwd()
    else:
        search_path = Path(path)
    
    if not search_path.exists():
        raise FileNotFoundError(f"Path does not exist: {search_path}")
    
    if not search_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {search_path}")
    
    # Load .gitignore patterns if provided
    gitignore_patterns = []
    if gitignore_path is not None:
        gitignore_path = Path(gitignore_path)
        if gitignore_path.exists():
            gitignore_patterns = _load_gitignore_patterns(gitignore_path)
    
    # Use pathlib's rglob for recursive file finding
    all_files = []
    all_directories = []

    for item in search_path.rglob("*"):
        if item.is_file():  # Only include files, not directories
            if not gitignore_patterns or \
                not _is_ignored(item, gitignore_patterns, search_path):
                all_files.append(item)
        elif item.is_dir():
            if not gitignore_patterns or \
                not _is_ignored(item, gitignore_patterns, search_path):
                all_directories.append(item)
    
    return all_files, all_directories

def _load_gitignore_patterns(gitignore_path: Path) -> List[str]:
    """Load and parse .gitignore patterns."""
    patterns = []
    
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle negated patterns (starting with !)
                if line.startswith('!'):
                    patterns.append(('include', line[1:]))
                else:
                    patterns.append(('exclude', line))
                    
    except Exception as e:
        print(f"Warning: Could not read .gitignore file {gitignore_path}: {e}")
    
    return patterns

def _is_ignored(path: Path, patterns: List[str], base_path: Path) -> bool:
    """
    Check if a path should be ignored based on .gitignore patterns.
    
    Args:
        path: Path to check
        patterns: List of (action, pattern) tuples from .gitignore
        base_path: Base directory for relative path calculations
        
    Returns:
        True if path should be ignored
    """
    # Get relative path from base directory
    try:
        relative_path = path.relative_to(base_path)
    except ValueError:
        # Path is not relative to base_path, use absolute
        relative_path = path
    
    # Convert to string for pattern matching
    path_str = str(relative_path).replace('\\', '/')  # Normalize separators
    
    # Track if this path is explicitly included
    explicitly_included = False
    
    for action, pattern in patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            if path.is_dir() and _matches_pattern(path_str, pattern[:-1]):
                return action == 'exclude'
        
        # Handle file patterns
        elif _matches_pattern(path_str, pattern):
            if action == 'include':
                explicitly_included = True
            else:  # exclude
                return True
    
    # If explicitly included, don't ignore
    if explicitly_included:
        return False
    
    # Check if any parent directory is ignored
    for parent in path.parents:
        if parent == base_path:
            break
        try:
            parent_relative = parent.relative_to(base_path)
            parent_str = str(parent_relative).replace('\\', '/')
            
            for action, pattern in patterns:
                if pattern.endswith('/') and \
                    _matches_pattern(parent_str, pattern[:-1]):
                    return action == 'exclude'
        except ValueError:
            continue
    
    return False

def _matches_pattern(path_str: str, pattern: str) -> bool:
    """Check if a path matches a gitignore pattern."""
    # Handle special cases
    if pattern == '*':
        return True
    
    # Convert gitignore pattern to fnmatch pattern
    fnmatch_pattern = pattern
    
    # Handle ** (recursive matching)
    if '**' in pattern:
        fnmatch_pattern = pattern.replace('**', '*')
    
    # Handle leading slash (match from root)
    if pattern.startswith('/'):
        fnmatch_pattern = pattern[1:]
        return fnmatch.fnmatch(path_str, fnmatch_pattern)
    
    # Handle trailing slash (directory only)
    if pattern.endswith('/'):
        fnmatch_pattern = pattern[:-1]
    
    # Check if pattern matches the path or any part of it
    path_parts = path_str.split('/')
    
    for i in range(len(path_parts)):
        test_path = '/'.join(path_parts[i:])
        if fnmatch.fnmatch(test_path, fnmatch_pattern):
            return True
    
    return False

# Example usage and test function
def test_get_files_list_recursively():
    """Test the function with various scenarios."""
    
    # Test 1: Basic usage with current directory
    print("=== Test 1: Current directory ===")
    files = get_files_list_recursively()
    print(f"Found {len(files)} files in current directory")
    for file in files[:5]:  # Show first 5 files
        print(f"  {file}")
    
    # Test 2: Specific directory
    print("\n=== Test 2: Specific directory ===")
    test_dir = Path(__file__).parent  # Directory containing this script
    files = get_files_list_recursively(test_dir)
    print(f"Found {len(files)} files in {test_dir}")
    
    # Test 3: With .gitignore (if it exists)
    print("\n=== Test 3: With .gitignore ===")
    gitignore_path = Path.cwd() / '.gitignore'
    if gitignore_path.exists():
        files = get_files_list_recursively(gitignore_path=gitignore_path)
        print(f"Found {len(files)} files (respecting .gitignore)")
    else:
        print("No .gitignore found in current directory")
    
    # Test 4: Error handling
    print("\n=== Test 4: Error handling ===")
    try:
        get_files_list_recursively("/nonexistent/path")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    try:
        get_files_list_recursively(__file__)  # Pass a file instead of directory
    except NotADirectoryError as e:
        print(f"Expected error: {e}")

if __name__ == "__main__":
    test_get_files_list_recursively()
