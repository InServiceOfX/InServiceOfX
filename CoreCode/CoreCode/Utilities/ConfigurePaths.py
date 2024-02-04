from dataclasses import dataclass
from pathlib import Path

@dataclass
class BasicProjectPaths:
    configure_paths_path: Path
    core_code_python_path: Path
    core_code_path: Path
    project_path: Path

def _setup_paths():
    """
    @fn _setup_paths
    @brief Auxiliary function to set up configure.py, making project aware of
    its file directory paths or "position."
    """

    # https://docs.python.org/3/reference/import.html#file__
    # __file__ indicates pathname of file from which the module was loaded (if
    # loaded from a file), or pathname of shared library file for extension
    # modules loaded dynamically from a shared library.
    current_filepath = Path(__file__).resolve() # Resolve to the absolute path.

    # These values are dependent upon where this file, configure_paths.py, is placed.
    # It seems that parent "0" is counted as the (sub)directory containing
    # this file.
    number_of_parents_to_python_library_path = 1
    number_of_parents_to_library_path = 2
    number_of_parents_to_project_path = 3

    return BasicProjectPaths(
        current_filepath,
        current_filepath.parents[number_of_parents_to_python_library_path],
        current_filepath.parents[number_of_parents_to_library_path],
        current_filepath.parents[number_of_parents_to_project_path])

def default_path_to_env_file():
    return _setup_paths().project_path / ".env"