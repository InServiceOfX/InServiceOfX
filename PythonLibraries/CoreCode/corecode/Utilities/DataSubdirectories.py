from corecode.Utilities import LoadConfigurationFile

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataSubdirectories:
    Data: Path = field(init=False)
    Models: Path = field(init=False)
    ModelsDiffusion: Path = field(init=False)
    ModelsDiffusionLoRAs: Path = field(init=False)
    ModelsLLM: Path = field(init=False)
    Public: Path = field(init=False)
    PublicFinances: Path = field(init=False)

    """
    We do not define a __init__(..) function because then the fields would be
    positional arguments required when creating a class instance.
    """
    def __post_init__(self):
        self.Data = LoadConfigurationFile.load_configuration_file()['BASE_DATA_PATH']
        self.Models = self.Data / "Models"
        self.ModelsDiffusion = self.Models / "Diffusion"
        self.ModelsDiffusionLoRAs = self.ModelsDiffusion / "LoRAs"
        self.ModelsLLM = self.Models / "LLM"
        self.Public = self.Data / "Public"
        self.PublicFinances = self.Public / "Finances"