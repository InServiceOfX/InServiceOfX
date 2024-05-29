from pathlib import Path

def parse_build_configuration_file(build_file_path):
    configuration = {}

    if build_file_path.exists():
        with open(build_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Skip comments and empty lines
                if stripped_line.startswith('#') or not stripped_line:
                    continue

                if '=' in stripped_line:
                    key, value = stripped_line.split('=', 1)

                    if key.strip() == "DOCKER_IMAGE_NAME":
                        configuration['DOCKER_IMAGE_NAME'] = value.strip()

    return configuration


def parse_run_configuration_file(configuration_file_path):
    configuration = {}
    mount_paths = []

    if configuration_file_path.exists():
        with open(configuration_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Skip comments and empty lines
                if stripped_line.startswith('#') or not stripped_line:
                    continue

                if '=' in stripped_line:
                    key, value = stripped_line.split('=', 1)

                    if key.strip().startswith("MOUNT_PATH_") or \
                        key.strip().starts_with("MOUNT_PATH"):

                        print(value)
                        print(Path(value.split(':', 1)[0]).resolve())

                        if ':' in value and \
                            Path(value.split(':', 1)[0]).resolve().is_dir():
                            mount_paths.append(value)

                        else:
                            print(
                                f"Invalid or nonexistent path in file: {line.split(':', 1)[0]}")

    configuration['mount_paths'] = mount_paths
    return configuration
