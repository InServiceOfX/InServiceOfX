from pathlib import Path

def get_project_directory_path():
    # Get the current file's absolute path.
    current_filepath = Path(__file__).resolve()

    # Assume the directory structure has not changed.
    number_of_parents_to_project_path = 4

    return current_filepath.parents[number_of_parents_to_project_path]

def get_project_directory_path_recursive():
    current_filepath = Path(__file__).resolve()

    # Traverse up directories to find one containing this unique identifier.
    for parent in current_filepath.parents:
        # Check if '.git' is in this directory
        if (parent / '.git').exists():
            return parent

    raise FileNotFoundError(
        "Repository main directory not found or .git wasn't.")