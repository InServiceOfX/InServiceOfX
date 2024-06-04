from corecode.FileIO import get_project_directory_path
from morediffusers.Configurations import LoRAsConfiguration
from pathlib import Path
import re
import yaml

class LoRAsConfigurationForMoreInstantID(LoRAsConfiguration):
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "ThirdParties" / \
                "MoreInstantID" / "loras_configuration.yml"
        ):
        super().__init__(configuration_path=configuration_path)
