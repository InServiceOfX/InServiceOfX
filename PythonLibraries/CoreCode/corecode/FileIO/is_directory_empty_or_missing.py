from pathlib import Path

def is_directory_empty_or_missing(directory_path: str) -> bool:
    path = Path(directory_path)
    return not path.exists() or not any(path.iterdir())
