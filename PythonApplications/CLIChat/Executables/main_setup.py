from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import yes_no_dialog, input_dialog
import shutil
import sys
from importlib.resources import files

def get_configuration_directory():
    configuration_dir = Path.home() / ".config" / "clichat" / "Configurations"
    
    if not configuration_dir.exists():
        create = yes_no_dialog(
            title="Create Configuration Directory",
            text=f"CLIChat needs to create its configuration directory at:\n{configuration_dir}\nProceed?"
        ).run()
        
        if create:
            configuration_dir.mkdir(parents=True, exist_ok=True)
            return configuration_dir
        else:
            print("Setup cancelled")
            sys.exit(1)
    else:
        print(f"Configuration directory exists: {configuration_dir}")

    return configuration_dir

def get_data_directory():
    result = input_dialog(
        title="Data Directory Selection",
        text=(
            "Enter path for CLIChat data storage\n(chat history, system "
            "messages, etc.)"),
        default=str(Path.cwd() / "clichat")
    ).run()
    
    if result is None:  # User pressed Cancel
        print("Setup cancelled")
        sys.exit(1)
        
    path = Path(result).expanduser().resolve()
    
    if not path.exists():
        create = yes_no_dialog(
            title="Create Data Directory",
            text=f"Directory {path} doesn't exist. Create it?"
        ).run()
        
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception as e:
                print(f"Failed to create data directory: {e}")
                sys.exit(1)
        else:
            print("Setup cancelled")
            sys.exit(1)
            
    return path

def get_package_directory():
    """Get the directory where the clichat package is installed."""
    try:
        # This will find the installed package location
        return files('clichat')
    except ImportError:
        # Fallback for development
        return Path(__file__).parent

def copy_configuration_files(
    configuration_dir: Path, data_dir: Path, package_dir: Path):
    """
    Copy from package_dir and into configuration_dir and data_dir.
    """
    # Create subdirectories
    data_dir = data_dir / "Data"
    configuration_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # Copy configuration files
    shutil.copy(
        package_dir / "Configurations" / "clichat_configuration.yml",
        configuration_dir / "clichat_configuration.yml"
    )

    shutil.copy(
        package_dir / "Data" / "chat_history.txt.example",
        data_dir / "chat_history.txt"
    )

    shutil.copy(
        package_dir / "Data" / "empty_system_messages.json",
        data_dir / "system_messages.json"
    )
    
    env_example = package_dir / "Configurations" / ".env.example"
    shutil.copy(env_example, configuration_dir / ".env")

def update_configuration_paths(
    configuration_path: Path, data_dir: Path):
    with open(configuration_path, 'r') as f:
        configuration_content = f.read()
    
    # Update paths in configuration
    configuration_content = configuration_content.replace(
        "Data/chat_history.txt",
        str(data_dir / "chat_history.txt")
    )

    configuration_content = configuration_content.replace(
        "Data/system_messages.json",
        str(data_dir / "system_messages.json")
    )
    
    with open(configuration_path, 'w') as f:
        f.write(configuration_content)

def setup_api_key(configuration_dir: Path):
    env_path = configuration_dir / ".env"
    
    result = input_dialog(
        title="Groq API Key Setup",
        text=(
            "Enter your Groq API key\n"
            "(Leave empty to configure manually later)\n"
            "Get your key at: https://console.groq.com/keys"
        ),
        password=True  # Hide API key while typing
    ).run()
    
    if result and result.strip():
        try:
            # Read existing .env content
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            # Replace the placeholder with actual API key
            env_content = env_content.replace(
                "GROQ_API_KEY=gsk_xxxxxx",
                f"GROQ_API_KEY={result.strip()}")
            
            # Write updated content
            with open(env_path, 'w') as f:
                f.write(env_content)
                
            print(f"\nAPI key configured successfully!")
        except Exception as e:
            print(f"\nFailed to update API key: {e}")
    
    print(f"\nEnvironment file location: {env_path}")
    print("You can manually edit this file anytime to update your API key.")

def main_setup():
    print("Welcome to CLIChat Setup")
    
    # Setup config directory in ~/.config/clichat
    configuration_dir = get_configuration_directory()
    print(f"Sanity check: Configuration directory: {configuration_dir}")

    # Get user's preferred data directory
    data_dir = get_data_directory()
    print(f"Sanity check: Data directory: {data_dir}")

    # Get package directory using the new method
    package_dir = get_package_directory()
    print(f"Sanity check: Package directory: {package_dir}")

    # Copy files
    copy_configuration_files(configuration_dir, data_dir, package_dir)
    
    # Setup API key
    setup_api_key(configuration_dir)
    
    # Update paths in the configuration file
    config_file_path = configuration_dir / "clichat_configuration.yml"
    update_configuration_paths(config_file_path, data_dir / "Data")
    
    print(f"\nSetup complete!")
    print(f"Configuration directory: {configuration_dir}")
    print(f"Data directory: {data_dir}")
    print(
        (
            "Please edit .env file in the Configurations directory with your "
            "API keys in the future."))

if __name__ == "__main__":
    main_setup()