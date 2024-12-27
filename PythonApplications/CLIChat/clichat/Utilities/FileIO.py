from pathlib import Path

def setup_chat_history_file(
    filepath: str | Path | None = None, 
    filename: str = "chat_history.txt") -> Path:
    """
    Sets up a chat history file in the specified directory or current working
    directory. Preserves relative paths if they exist.
    
    Args:
        filepath: Path to directory or file. If None, uses current working
        directory
        filename: Name of the chat history file to create. Defaults to
        chat_history.txt
        
    Returns:
        Path object pointing to the directory containing the chat history
        file. Keeps relative path if input was relative.
    """
    if filepath is None:
        history_file = Path.cwd() / filename
        if not history_file.exists():
            history_file.touch()
        return history_file
        
    # Convert to Path object but preserve relative/absolute state
    filepath = Path(filepath)
    was_relative = not filepath.is_absolute()
    
    # Convert to absolute for existence check
    abs_filepath = filepath if filepath.is_absolute() else Path.cwd() / filepath
    
    # If filepath points to a file, get its parent directory
    target_dir = abs_filepath.parent if abs_filepath.is_file() else abs_filepath

    if not target_dir.exists():
        target_dir = Path.cwd()

    # Create file if it doesn't exist
    history_file = target_dir / filename
    if not history_file.exists():
        history_file.touch()
    
    # Return relative path if input was relative
    if was_relative:
        try:
            return history_file.relative_to(Path.cwd())
        except ValueError:
            return history_file
    return history_file

def get_chat_history_path(configuration):
    """
    Sets up and validates chat history path in configuration.
    Handles both relative and absolute paths.
    """
    if configuration.chat_history_path is None:
        result_path = setup_chat_history_file()
        configuration.chat_history_path = str(result_path)
        return result_path
    
    # Convert to Path while preserving relative/absolute state
    history_path = Path(configuration.chat_history_path)
    abs_history_path = history_path if history_path.is_absolute() else \
        Path.cwd() / history_path

    # TODO: Is it better to check if the parent directory exists?
    if not abs_history_path.exists():
        # File doesn't exist yet so we can't use is_file(). Instead, check the
        # following:
        if bool(abs_history_path.suffix) or '.' in abs_history_path.name:
            result_path = setup_chat_history_file(
                history_path.parent,
                history_path.name)
            configuration.chat_history_path = str(result_path)
            return result_path
        else:
            result_path = setup_chat_history_file(history_path)
            return result_path

    return history_path

def get_existing_chat_history_path_or_fail(configuration):
    if configuration.chat_history_path is None or \
        configuration.chat_history_path == "":
        return None

    # Convert to Path while preserving relative/absolute state
    history_path = Path(configuration.chat_history_path)
    abs_history_path = history_path if history_path.is_absolute() else \
        Path.cwd() / history_path

    if not abs_history_path.exists():
        raise FileNotFoundError(
            f"Chat history file does not exist: {abs_history_path}")

    return history_path

def get_path_from_configuration(configuration, field):
    """
    Raises FileNotFoundError if file path does not exist for the given field
    value.
    """
    if getattr(configuration, field) is None:
        return None

    path = Path(getattr(configuration, field))
    abs_path = path if path.is_absolute() else Path.cwd() / path

    if not abs_path.exists():
        raise FileNotFoundError(f"File does not exist: {abs_path}")

    return abs_path
