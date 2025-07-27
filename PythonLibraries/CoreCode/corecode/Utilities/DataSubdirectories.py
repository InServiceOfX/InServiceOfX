from corecode.Configuration import LoadConfigurationFile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from warnings import warn

@dataclass
class DataSubdirectories:
    Data: Path = field(init=False)
    
    DataPaths: List[Path] = field(init=False, default_factory=list)
    
    Models: Path = field(init=False)
    ModelsDiffusion: Path = field(init=False)
    ModelsDiffusionLoRAs: Path = field(init=False)
    ModelsLLM: Path = field(init=False)
    Public: Path = field(init=False)
    PublicFinances: Path = field(init=False)

    PromptsCollection: Path = field(init=False)

    """
    We do not define a __init__(..) function because then the fields would be
    positional arguments required when creating a class instance.
    """
    def __post_init__(self):
        # Load configuration
        config = LoadConfigurationFile.load_configuration_file()
        
        # Backward compatibility - set primary Data path
        self.Data = config['BASE_DATA_PATH']
        if isinstance(self.Data, str):
            self.Data = Path(self.Data)

        # Collect all BASE_DATA_PATH_X keys
        self.DataPaths = []
        
        # Add the primary data path first
        self.DataPaths.append(self.Data)
        
        # Find all numbered data paths
        for key, value in config.items():
            if key.startswith('BASE_DATA_PATH_') and key != 'BASE_DATA_PATH':
                # Extract the number from the key
                try:
                    # Remove 'BASE_DATA_PATH_' prefix to get the number
                    number_str = key.replace('BASE_DATA_PATH_', '')
                    int(number_str)

                    # Add to DataPaths list
                    self.DataPaths.append(value)
                    
                except ValueError:
                    # Skip if the suffix is not a valid integer
                    continue
        
        # Create standard subdirectories from primary Data path (backward compatibility)
        self.Models = self.Data / "Models"
        self.ModelsDiffusion = self.Models / "Diffusion"
        self.ModelsDiffusionLoRAs = self.ModelsDiffusion / "LoRAs"
        self.ModelsLLM = self.Models / "LLM"
        self.Public = self.Data / "Public"
        self.PublicFinances = self.Public / "Finances"

        if config['PROMPTS_COLLECTION_PATH'] is not None:
            self.PromptsCollection = Path(config['PROMPTS_COLLECTION_PATH'])
        elif config['PROMPTS_COLLECTION_PATH'] is None:
            relative_prompts_collection_path = "Prompts/PromptsCollection"
            for path in self.DataPaths:
                if (path / relative_prompts_collection_path).exists():
                    self.PromptsCollection = \
                        path / relative_prompts_collection_path
                    break
            if self.PromptsCollection is None:
                warn(
                    f"Prompts collection path not found in {self.DataPaths}")
                self.PromptsCollection = \
                    self.Data / "Prompts" / "PromptsCollection"
        else:
            raise ValueError(
                f"Invalid prompts collection path: {config['PROMPTS_COLLECTION_PATH']}")

    def get_data_path(self, index: int = 0) -> Path:
        """
        Get a specific data path by index.
        
        Args:
            index: Index of the data path (0 = primary, 1 = BASE_DATA_PATH_1, etc.)
            
        Returns:
            Path object for the requested data path
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.DataPaths):
            raise IndexError(
                f"Data path index {index} out of range. Available: 0-{len(self.DataPaths)-1}")
        return self.DataPaths[index]
        
    def create_subdirectories_for_path(self, data_path_index: int = 0) -> 'DataSubdirectories':
        """
        Create a new DataSubdirectories instance using a specific data path.
        
        Args:
            data_path_index: Index of the data path to use
            
        Returns:
            New DataSubdirectories instance with subdirectories based on the specified path
        """
        data_path = self.get_data_path(data_path_index)
        
        # Create a new instance
        new_instance = DataSubdirectories()
        
        # Set the primary data path
        new_instance.Data = data_path
        # Only this one path
        new_instance.DataPaths = [data_path]
        
        # Create subdirectories based on this path
        new_instance.Models = data_path / "Models"
        new_instance.ModelsDiffusion = new_instance.Models / "Diffusion"
        new_instance.ModelsDiffusionLoRAs = new_instance.ModelsDiffusion / "LoRAs"
        new_instance.ModelsLLM = new_instance.Models / "LLM"
        new_instance.Public = data_path / "Public"
        new_instance.PublicFinances = new_instance.Public / "Finances"
        
        return new_instance

def setup_datasets_path():
    data_subdirectories = DataSubdirectories()

    data_path = data_subdirectories.get_data_path(0)
    datasets_path = data_path / "Datasets"

    if not datasets_path.exists():
        for path in data_subdirectories.DataPaths:
            datasets_path = Path(path) / "Datasets"
            if datasets_path.exists():
                break
        if not datasets_path.exists():
            raise FileNotFoundError(
                f"Datasets path not found in {data_subdirectories.DataPaths}")

    return datasets_path