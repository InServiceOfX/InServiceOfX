class ReadBuildConfiguration:

    def __init__(self, required_keys):
        """
        Args:
            required_keys (list): List of required keys.

        Here, you'll need to specify any and all required keys; no keys are
        assumed to be required in any way. So make sure to include for almost
        all typical Docker builds:
        - BASE_IMAGE
        - DOCKER_IMAGE_NAME
        """
        self.required_keys = required_keys

    def read_build_configuration(self, config_path):
        """
        TODO: Replace use of this function in CommonUtilities.py.

        Reads the build_configuration.txt file and parses parameters.

        Args:
            config_path (Path): Path to the build_configuration.txt file.

        Returns:
            dict: Dictionary containing the extracted parameters.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If any required parameter is missing.
        """
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file '{config_path}' does not exist.")

        configuration = {}

        with config_path.open('r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                if '=' not in line:
                    continue  # Skip lines without key=value format

                key, value = line.split('=', 1)
                key = key.strip().upper()
                value = value.strip()
                if key in self.required_keys:
                    configuration[key] = value

        missing_keys = self.required_keys - configuration.keys()
        if missing_keys:
            raise ValueError(
                f"Missing required configuration parameters: {', '.join(missing_keys)}")

        return configuration

class ReadBuildConfigurationWithNVIDIAGPU(ReadBuildConfiguration):
    def __init__(self):
        required_keys = {
            "ARCH",
            "PTX",
            "COMPUTE_CAPABILITY",
            "DOCKER_IMAGE_NAME",
            "BASE_IMAGE"}

        super().__init__(required_keys)

class ReadBuildConfigurationForMinimalStack(ReadBuildConfiguration):
    def __init__(self):
        required_keys = {
            "DOCKER_IMAGE_NAME",
            "BASE_IMAGE"}

        super().__init__(required_keys)
