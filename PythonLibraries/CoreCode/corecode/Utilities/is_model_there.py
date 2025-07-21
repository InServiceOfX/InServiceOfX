from .DataSubdirectories import DataSubdirectories
from pathlib import Path

def is_model_there(
    relative_model_path: str,
    data_subdirectories = None) -> bool:
    if data_subdirectories is None:
        data_subdirectories = DataSubdirectories()

    is_model_downloaded = False
    model_path = None

    for path in data_subdirectories.DataPaths:
        path = Path(path)
        if (path / relative_model_path).exists():
            is_model_downloaded = True
            model_path = path / relative_model_path
            break

    return is_model_downloaded, model_path