from .ConfigurationData import ConfigurationData
from corecode.FileIO import get_default_path_to_config_file
from pathlib import Path

import re

class LoadConfigurationFile:
    
    @staticmethod
    def _parse_configuration_file(file_object):
        config_data = ConfigurationData()

        # Regex pattern to match BASE_DATA_PATH_X where X is any number
        numbered_path_pattern = re.compile(r'^BASE_DATA_PATH_(\d+)$')
        
        for line in file_object:
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue

            stripped_line = line.strip()

            # Split line into key and value
            if '=' in stripped_line:
                key, value = stripped_line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove surrounding double quotes from value if present.
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                if key == 'BASE_DATA_PATH':
                    config_data.BASE_DATA_PATH = value
                elif key == 'PROMPTS_COLLECTION_PATH':
                    config_data.PROMPTS_COLLECTION_PATH = value \
                        if value else None
                elif numbered_path_pattern.match(key):
                    # Handle any BASE_DATA_PATH_X where X is a number
                    match = numbered_path_pattern.match(key)
                    number = match.group(1)
                    config_data.add_numbered_data_path(number, value)

        return config_data

    @staticmethod
    def load_configuration_file(file_path: str | Path | None = None):
        """
        @return configuration, Python dict with key name to str for paths,
        strings for other values.
        """
        configuration = {}

        if file_path is None:
            file_path = get_default_path_to_config_file()
        elif isinstance(file_path, str):
            file_path = Path(file_path)

        with open(file_path, 'r') as file:
            configuration = LoadConfigurationFile._parse_configuration_file(
                file)
        return configuration.to_dict()
