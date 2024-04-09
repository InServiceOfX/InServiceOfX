from pathlib import Path

def get_main_directory_path():
    """
    TODO: This is a copy of, in Utilities/ConfigurePaths.py, of _setup_paths.
    """
    # Get the current file's absolute path.
    current_filepath = Path(__file__).resolve()

    # Assume the directory structure has not changed.
    number_of_parents_to_project_path = 3

    return current_filepath.parents[number_of_parents_to_project_path]

def get_main_directory_path_recursive():
    current_file_path = Path(__file__).resolve()

    # Traverse up directories to find one containing this unique identifier.
    for parent in current_filepath.parents:
        # Check if '.git' is in this directory
        if (parent / '.git').exists():
            return parent

    raise FileNotFoundError(
        "Repository main directory not found or .git wasn't.")