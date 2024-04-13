from corecode.Utilities import LoadConfigurationFile

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataSubdirectories:
    Models: Path = field(init=False)
    ModelsDiffusion: Path = field(init=False)

    """
    We do not define a __init__(..) function because then the fields would be
    positional arguments required when creating a class instance.
    """
    def __post_init__(self):
        self.Models = \
            LoadConfigurationFile.load_configuration_file()['BASE_DATA_PATH'] / "Models"

        self.ModelsDiffusion = self.Models / "Diffusion"