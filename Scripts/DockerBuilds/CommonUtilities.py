from pathlib import Path

def get_project_directory():
    # Resolve to absolute path.
    current_filepath = Path(__file__).resolve()

    # This variable's value depends on the location of this file relative to
    # other subdirectories. It depends on the location of this file, even if
    # this function is imported and used in another file in another
    # subdirectory.
    parent_subdirectories = 2

    project_directory = current_filepath.parents[parent_subdirectories]

    if not project_directory.is_dir():
        print(f"{project_directory} is not a directory")
        exit(1)

    if not project_directory.exists():
        print(f"{project_directory} is not an existing directory")
        exit(1)

    return project_directory


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


class DefaultValues:
    _BUILD_FILE_NAME="build_docker_configuration.txt"
    _CONFIGURATION_FILE_NAME="run_docker_configuration.txt"

    @classmethod
    @property
    def BUILD_FILE_NAME(cls):
        return cls._BUILD_FILE_NAME

    @classmethod
    @property 
    def CONFIGURATION_FILE_NAME(cls):
        return cls._CONFIGURATION_FILE_NAME


class CreateDockerRunCommand:
    def __init__(self, project_directory, build_configuration, configuration):

        self.docker_image_name = build_configuration["DOCKER_IMAGE_NAME"]
        self.mount_paths = configuration["mount_paths"]

        self.docker_run_command = self.create_docker_run_command(
            project_directory,
            self.mount_paths,
            self.docker_image_name)


    def create_docker_run_command(
        self,
        project_directory,
        mount_paths,
        DOCKER_IMAGE_NAME
        ):
        # Run command
        # -it - i stands for interactive, so this flag makes sure that standard
        # input ('STDIN') remains open even if you're not attached to container.
        # -t stands for pseudo-TTY, allocates a pseudo terminal inside
        # container, used to make environment inside container feel like a
        # regular shell session.
        docker_run_command = f"docker run -v {project_directory}:/InServiceOfX --gpus all -it "

        # Add mount paths from configuration file
        for mount_path in mount_paths:
            docker_run_command += f"-v {mount_path} "

        # -e flag sets environment and enables CUDA Forward Compatibility instead of
        # default CUDA Minor Version Compatibility.
        docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

        # Add the port 7860 for gradio applications.
        docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host "

        docker_run_command += DOCKER_IMAGE_NAME

        print(docker_run_command)

        return docker_run_command