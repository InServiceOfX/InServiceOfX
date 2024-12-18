from pathlib import Path

def setup_chat_history_file(
    filepath: str | Path | None = None, 
    filename: str = "chat_history.txt") -> Path:
    """
    Sets up a chat history file in the specified directory or current working
    directory.
    
    Args:
        filepath: Path to directory or file. If None, uses current working
        directory
        filename: Name of the chat history file to create. Defaults to
        chat_history.txt
        
    Returns:
        Path object pointing to the directory containing the chat history
        file
    """
    # Handle None filepath
    if filepath is None:
        target_dir = Path.cwd()
    else:
        # Convert to Path object
        filepath = Path(filepath)
        
        # If filepath points to a file, get its parent directory
        target_dir = filepath.parent if filepath.is_file() else filepath
        
        # If directory doesn't exist, use current working directory
        if not target_dir.exists():
            target_dir = Path.cwd()
    
    # Construct full path to chat history file
    history_file = target_dir / filename
    
    # Create file if it doesn't exist
    if not history_file.exists():
        history_file.touch()
    
    return target_dir

def get_chat_history_path(configuration):
    """
    Sets up and validates chat history path in configuration.
    If path is None/empty or invalid, creates new path using
    setup_chat_history_file.
    Mutates the configuration object with validated path.
    """
    # If path is None (including empty string converted to None by validator)
    if configuration.chat_history_path is None:
        target_dir = setup_chat_history_file()
        configuration.chat_history_path = str(target_dir / "chat_history.txt")
        return target_dir / "chat_history.txt"
    
    # Check if existing path is valid
    history_path = Path(configuration.chat_history_path)
    if not history_path.parent.exists():
        # Invalid path, create new one and update configuration
        target_dir = setup_chat_history_file(history_path)
        configuration.chat_history_path = str(target_dir / "chat_history.txt")
        return target_dir / "chat_history.txt"

    return history_path