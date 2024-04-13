from corecode.FileIO import get_default_path_to_config_file

from pathlib import Path

class LoadConfigurationFile:

	key_parser_map = {
		'BASE_DATA_PATH': lambda value: Path(value)
	}	
	
	@staticmethod
	def _parse_configuration_file(file_object, configuration):
		for line in file_object:

			# Trim leading whitespace and check for comments or empty lines
			stripped_line = line.strip()
			if stripped_line.startswith('#') or not stripped_line:
				continue

			# Split line into key and value
			if '=' in stripped_line:
				key, value = stripped_line.split('=', 1)

				key = key.strip()
				value = value.strip()

				# Remove surrounding double quotes from value if present.
				if value.startswith('"') and value.endswith('"'):
					value = value[1:-1]

				if key in LoadConfigurationFile.key_parser_map:
					try:
						value = LoadConfigurationFile.key_parser_map[key](value)
					except Exception as err:
						print(f"Error processing {key}: {err}")

				configuration[key] = value

		return configuration

	@staticmethod
	def load_configuration_file():
		"""
		@return configuration, Python dict with key name to pathlib.Path, e.g.
		'BASE_DATA_PATH': PosixPath('/Data')
		"""
		configuration = {}
		with open(get_default_path_to_config_file(), 'r') as file:

			configuration = LoadConfigurationFile._parse_configuration_file(
				file,
				configuration)
		return configuration