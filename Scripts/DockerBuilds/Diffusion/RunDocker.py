"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py
"""

from pathlib import Path
import os, sys

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from CommonUtilities import (
    get_project_directory,
    parse_build_configuration_file,
    parse_run_configuration_file)

# Global variables
BUILD_FILE_NAME="build_docker_configuration.txt"
CONFIGURATION_FILE_NAME="run_docker_configuration.txt"

def parse_mount_paths_file():
    mount_paths = []
    file_path = Path(__file__).resolve().parent / CONFIGURATION_FILE_NAME
    if file_path.exists():
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Skip comments and empty lines
                if stripped_line.startswith('#') or not stripped_line:
                    continue

                path_parts = stripped_line.split(':', 1)
                if ':' in stripped_line and \
                    Path(stripped_line.split(':', 1)[0]).resolve().is_dir():
                    mount_paths.append(stripped_line)
                else:
                    print(f"Invalid or nonexistent path in file: {line.split(':', 1)[0]}")
    return mount_paths

def main():
    project_directory = get_project_directory()

    # Path to the build configuration file.
    build_file_path = Path(__file__).resolve().parent / BUILD_FILE_NAME
    build_configuration = parse_build_configuration_file(build_file_path)

    # Path to the configuration file.
    configuration_file_path = Path(__file__).resolve().parent / \
        CONFIGURATION_FILE_NAME
    configuration = parse_run_configuration_file(configuration_file_path)

    DOCKER_IMAGE_NAME = build_configuration["DOCKER_IMAGE_NAME"]
    mount_paths = configuration["mount_paths"]

    # Run command
    # -it - i stands for interactive, so this flag makes sure that standard
    # input ('STDIN') remains open even if you're not attached to container.
    # -t stands for pseudo-TTY, allocates a pseudo terminal inside container,
    # used to make environment inside container feel like a regular shell
    # session.
    docker_run_command = f"docker run -v {project_directory}:/InServiceOfX --gpus all -it "

    # Add mount paths from configuration file
    for mount_path in mount_paths:
        docker_run_command += f"-v {mount_path} "

    # Check for additional mount paths from user input.
    for mount_path in sys.argv[1:]:
        if ':' in mount_path:
            host_path, container_path = mount_path.split(':', 1)
            if Path(host_path).is_dir():
                docker_run_command += f"-v {mount_path}:{container_path} "
            else:
                print(f"The path '{host_path}' is not an existing path.")
                continue
        else:
            print(f"Invalid mount path format: {mount_path}")

    # -e flag sets environment and enables CUDA Forward Compatibility instead of
    # default CUDA Minor Version Compatibility.
    docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

    # Add the port 7860 for gradio applications.
    docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host "

    # Originally,
    #docker_run_command += "--ulimit memlock=-1 "
    # But on my target, I do run
    # ulimit -a
    # and see what my system uses.
    #docker_run_command += "--ulimit memlock=1968848 "

    # https://www.tutorialspoint.com/setting-ulimit-values-on-docker-containers
    # Originally,
    #docker_run_command += "--ulimit stack=67108864 "
    # But on my target, I do run
    # ulimit -a
    # and see what my system uses.
    #docker_run_command += "--ulimit stack=8192 "

    docker_run_command += DOCKER_IMAGE_NAME

    print(docker_run_command)

    os.system(docker_run_command)


if __name__ == "__main__":
    main()