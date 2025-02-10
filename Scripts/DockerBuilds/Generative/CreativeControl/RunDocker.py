"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py [directory_path] [--arm64]
"""

from pathlib import Path
import os, sys

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from CommonUtilities import (
    DefaultValues,
    get_project_directory)

from Utilities import ReadBuildConfigurationForMinimalStack

class CreateDockerRunCommandForServerAndData:
    def __init__(
        self,
        project_directory,
        build_configuration,
        run_configuration):
        self.project_directory = project_directory
        self.docker_image_name = build_configuration["DOCKER_IMAGE_NAME"]
        self.ports = run_configuration["ports"]
        self.mount_paths = run_configuration["mount_paths"]

        self.docker_run_command = self.create_docker_run_command(
            project_directory,
            self.docker_image_name,
            run_configuration)

    def create_docker_run_command(
        self,
        project_directory,
        docker_image_name,
        run_configuration):

        # -it - i stands for interactive, so this flag makes sure that standard
        # input ('STDIN') remains open even if you're not attached to container.
        # -t stands for pseudo-TTY, allocates a pseudo terminal inside
        # container, used to make environment inside container feel like a
        # regular shell session.
        docker_run_command = \
            f"docker run -v {project_directory}:/InServiceOfX --gpus all -it "

        # Add mount paths from configuration file
        for mount_path in run_configuration["mount_paths"]:
            docker_run_command += f"-v {mount_path} "

        # -e flag sets environment and enables CUDA Forward Compatibility instead of
        # default CUDA Minor Version Compatibility.
        docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

        for port in run_configuration["ports"]:
            docker_run_command += f"-p {port}:{port} "

        docker_run_command += "--rm --ipc=host "

        docker_run_command += docker_image_name

        return docker_run_command

def parse_run_configuration_file(configuration_file_path):
    configuration = {}
    ports = []
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
                    key = key.strip()

                    # Handle mount paths
                    if key.startswith("EXPOSE_PORT_") or \
                        key.strip().startswith("EXPOSE_PORT"):
                        try:
                            port = int(value.strip())
                            ports.append(port)
                        except ValueError:
                            print(f"Invalid port number: {value}")
                    elif key.startswith("MOUNT_PATH_") or \
                        key.strip().startswith("MOUNT_PATH"):

                        if ':' in value and \
                            Path(value.split(':', 1)[0]).resolve().is_dir():
                            mount_paths.append(value)
                        else:
                            print(
                                f"Invalid or nonexistent path in file: {line.split(':', 1)[0]}")

    else:
        print(f"Configuration file '{configuration_file_path}' does not exist.")

    configuration['ports'] = ports
    configuration['mount_paths'] = mount_paths
    return configuration


def main():

    build_file_path = Path(__file__).resolve().parent / \
        DefaultValues.BUILD_FILE_NAME
    build_configuration = \
        ReadBuildConfigurationForMinimalStack().read_build_configuration(
            build_file_path)

    run_configuration_file_path = Path(__file__).resolve().parent / \
        DefaultValues.RUN_CONFIGURATION_FILE_NAME
    run_configuration = parse_run_configuration_file(
        run_configuration_file_path)

    create_docker_run_command = CreateDockerRunCommandForServerAndData(
        get_project_directory(),
        build_configuration,
        run_configuration)

    print(create_docker_run_command.docker_run_command)

    os.system(create_docker_run_command.docker_run_command)

if __name__ == "__main__":
    main()
