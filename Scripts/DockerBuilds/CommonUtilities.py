from pathlib import Path
import argparse
import os
import re
import sys
import subprocess


def run_command(command, cwd=None):
    """
    Runs a shell command and captures its output.

    Args:
        command (str): The command to execute.
        cwd (Path, optional): The working directory to execute the command in.

    Returns:
        subprocess.CompletedProcess: The result of the executed command.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            status.
    """
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            # Commented out to try to stream output directly to console.
            #check=True,
            text=True,
            cwd=cwd)
        if result.returncode != 0:
            print(
                f"Error: Command '{command}' failed with exit code {result.returncode}",
                file=sys.stderr)
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as err:
        if err.returncode == 126:
            print(
                f"Error: Permission denied. Cannot run the command: {command}",
                file=sys.stderr)
        else:
            print(
                f"Error: Command '{command}' failed with exit code {err.returncode}",
                file=sys.stderr)
        sys.exit(err.returncode)

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

def read_build_configuration(config_path):
    """
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
    required_keys = {"ARCH", "PTX", "COMPUTE_CAPABILITY", "DOCKER_IMAGE_NAME"}

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
            if key in required_keys:
                configuration[key] = value

    missing_keys = required_keys - configuration.keys()
    if missing_keys:
        raise ValueError(
            f"Missing required configuration parameters: {', '.join(missing_keys)}")

    return configuration

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

def get_docker_builds_directory():
    return get_project_directory() / "Scripts" / "DockerBuilds"

def concatenate_dockerfiles(output_dockerfile, *dockerfile_paths):
    """
    Concatenates multiple Dockerfile components into a single Dockerfile.

    Args:
        output_dockerfile (Path): Path where the concatenated Dockerfile will be
        saved.
        *dockerfile_paths (Path): Arbitrary number of paths to Dockerfiles to
        concatenate.

    Raises:
        FileNotFoundError: If any of the input Dockerfile components are
        missing.
    """
    with output_dockerfile.open('w') as outfile:
        for file_path in dockerfile_paths:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Dockerfile component '{file_path}' does not exist.")
            with file_path.open('r') as infile:
                outfile.write(infile.read())
# Ensure separation between files
                outfile.write('\n')

    print(f"Successfully concatenated Dockerfiles into '{output_dockerfile}'.")

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