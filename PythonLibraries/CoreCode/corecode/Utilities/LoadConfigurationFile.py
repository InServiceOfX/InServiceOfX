from corecode.FileIO import get_default_path_to_config_file
from pathlib import Path

class LoadConfigurationFile:

	# Define base patterns that should be parsed as paths
	path_patterns = {
		'BASE_DATA_PATH',  # Exact match
		'BASE_DATA_PATH_'  # Prefix for numbered paths
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

				# Apply appropriate parser
				parsed_value = LoadConfigurationFile._apply_parser(key, value)
				configuration[key] = parsed_value

		return configuration

	@staticmethod
	def _apply_parser(key, value):
		"""
		Apply the appropriate parser based on key patterns.
		"""
		# Check if key matches any path pattern
		if (key in LoadConfigurationFile.path_patterns or 
			key.startswith('BASE_DATA_PATH_')):
			try:
				return Path(value)
			except Exception as err:
				print(f"Error processing path {key}: {err}")
				return value
		
		# For all other keys, return as-is
		return value

	@staticmethod
	def load_configuration_file():
		"""
		@return configuration, Python dict with key name to pathlib.Path for
		paths, strings for other values.
		"""
		configuration = {}
		with open(get_default_path_to_config_file(), 'r') as file:

			configuration = LoadConfigurationFile._parse_configuration_file(
				file,
				configuration)
		return configuration