"""
USAGE:
Usage examples:

python run_load_and_save_dataset.py pisterlabs/promptset
python run_load_and_save_dataset.py OpenAssistant/oasst1
"""
from pathlib import Path
import argparse
import sys

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_transformers_directory = \
    python_libraries_path / "HuggingFace" / "MoreTransformers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_transformers_directory) in sys.path:
    sys.path.append(str(more_transformers_directory))

from corecode.Utilities import setup_datasets_path
from moretransformers.Wrappers.Datasets import LoadAndSaveLocally

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and save datasets from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_and_save_dataset.py squad
  python load_and_save_dataset.py pisterlabs/promptset
  python load_and_save_dataset.py OpenAssistant/oasst1
        """
    )
    
    parser.add_argument(
        'dataset_name',
        type=str,
        help='Name of the dataset to load (e.g., "squad", "pisterlabs/promptset")'
    )
    
    return parser.parse_args()

def check_save_path_exists_and_nonempty(load_and_save_locally):
    """
    Check if the save path exists and is non-empty.
    
    Args:
        load_and_save_locally: LoadAndSaveLocally instance
        
    Returns:
        bool: True if path exists and is non-empty, False otherwise
    """
    try:
        save_path = load_and_save_locally._get_save_path()
        
        # Check if path exists
        if not save_path.exists():
            return False
        
        # Check if path is non-empty (has files or subdirectories)
        try:
            # Check if directory has any contents
            has_contents = any(save_path.iterdir())
            return has_contents
        except (PermissionError, OSError):
            # If we can't read the directory, assume it's non-empty
            return True
            
    except Exception as e:
        print(f"Warning: Could not check save path: {e}")
        return False

if __name__ == "__main__":
    args = parse_arguments()
    
    dataset_name = args.dataset_name
    
    print(f"Loading dataset: {dataset_name}")
    
    try:
        datasets_path = setup_datasets_path()
        print(f"Datasets path: {datasets_path}")
        print(f"Path exists: {datasets_path.exists()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    
    # Check if save path exists and is non-empty before loading
    if check_save_path_exists_and_nonempty(load_and_save_locally):
        save_path = load_and_save_locally._get_save_path()
        print(
            f"Warning: Save path already exists and is non-empty: {save_path}")
        print(
            "Skipping dataset loading and saving to avoid overwriting existing data.")
        print(
            "If you want to overwrite, please remove or rename the existing directory first.")
        sys.exit(0)
    
    dataset = load_and_save_locally.load_dataset(dataset_name)
    if dataset:
        print(f"Successfully loaded dataset: {dataset_name}")
    else:
        print(f"Failed to load dataset: {dataset_name}")
        sys.exit(1)

    load_and_save_locally.save_to_disk()
    print(f"Successfully saved dataset: {load_and_save_locally._dataset_name}")