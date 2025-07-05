import warnings
from typing import Any, Optional
from pathlib import Path

try:
    import datasets
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import list_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn(
        (
            "datasets is not installed. Please install it with `pip install `"
            "`datasets`"))

class LoadAndSaveLocally:
    def __init__(self, save_path: str | Path | None = None):
        self._data = None
        self._dataset_name = None
        self._available_datasets = None

        if save_path is None:
            save_path = str(Path.cwd())
        self._save_path = str(save_path)

    def get_available_datasets(self):
        if not DATASETS_AVAILABLE:
            warnings.warn("datasets library not available")
            return None
        self._available_datasets = list_datasets()
        return self._available_datasets

    def is_dataset_available(self, dataset_name: str) -> bool:
        if not DATASETS_AVAILABLE:
            warnings.warn("datasets library not available")
            return False
        if self._available_datasets is None:
            warnings.warn(
                "available datasets not availablej; run get_available_datasets")
            return False
        return any(element.id == dataset_name \
            for element in self._available_datasets)

    def _parse_dataset_name(self, dataset_name: Optional[str]) -> Path:
        """
        Parse dataset name and return the appropriate save/load path.
        
        Args:
            dataset_name: Name like "pisterlabs/promptset", "/promptset", or
            "promptset"
            
        Returns:
            Path: The full path where the dataset should be saved/loaded
        """
        if dataset_name is None:
            return self._save_path / "dataset"
        
        # Split by "/" to handle organization/repo format
        parts = dataset_name.split("/")
        # Has organization (e.g.,"pisterlabs/promptset")        
        if len(parts) > 1 and parts[0]:
            # Create organization directory and use repo name
            org_dir = Path(self._save_path) / parts[0]
            org_dir.mkdir(exist_ok=True)
            final_name = parts[1]
            return org_dir / final_name
        else:
            # No organization or empty organization (e.g., "promptset" or
            # "/promptset")
            final_name = parts[-1] if parts else dataset_name
            return Path(self._save_path) / final_name

    def load_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        **kwargs):
        self._dataset_name = dataset_name
        try:
            if split is not None:
                # Add split to kwargs if it's provided
                kwargs['split'] = split
            
            # Call load_from_disk with kwargs (which may be empty)
            if kwargs:
                self._data = load_dataset(dataset_name, **kwargs)
            else:
                self._data = load_dataset(dataset_name)
            return self._data
            
        except Exception as e:
            warnings.warn(f"Failed to load dataset from {dataset_name}: {e}")
            return self._data

    def _get_save_path(self, dataset_name: Optional[str] = None) -> Path:
        if self._save_path is None:
            warnings.warn("save_path is not set")
            return None
        if self._dataset_name is not None:
            dataset_name = self._dataset_name

        return self._parse_dataset_name(dataset_name)

    def save_to_disk(self, dataset_name: Optional[str] = None, **kwargs) -> bool:
        """
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not DATASETS_AVAILABLE:
                warnings.warn("datasets library not available")
                return False

            save_path = self._get_save_path(dataset_name)

            # Call datasets save_to_disk
            if kwargs:
                self._data.save_to_disk(str(save_path), **kwargs)
            else:
                self._data.save_to_disk(str(save_path))
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to save dataset to disk: {e}")
            return False

    def load_from_disk(
        self,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs) -> Optional[Any]:
        """
        Returns:
            Loaded dataset or None if failed
        """
        try:
            if not DATASETS_AVAILABLE:
                warnings.warn("datasets library not available")
                return None
            
            if self._save_path is None:
                warnings.warn("save_path is not set")
                return None
            
            if self._dataset_name is not None:
                dataset_name = self._dataset_name

            load_path = self._parse_dataset_name(dataset_name)

            if split is not None:
                kwargs['split'] = split
            
            if kwargs:
                return load_from_disk(str(load_path), **kwargs)
            else:
                return load_from_disk(str(load_path))
            
        except Exception as e:
            warnings.warn(f"Failed to load dataset from {load_path}: {e}")
            return None


