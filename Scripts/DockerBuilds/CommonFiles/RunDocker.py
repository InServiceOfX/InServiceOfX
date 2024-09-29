"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py [directory_path]
"""

from pathlib import Path
import os, sys

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from CommonUtilities import (
    DefaultValues,
    get_docker_builds_directory,
    get_project_directory,
    parse_run_configuration_file,
    read_build_configuration)


def main():
    docker_builds_directory = get_docker_builds_directory()

    # Check for command-line argument for directory path
    if len(sys.argv) > 1:
        dir_path = Path(sys.argv[1]).resolve()
    else:
        # Default to the specific directory structure
        dir_path = docker_builds_directory / "LLM" / "Meta" / "FullLlama"

    # Validate the directory path
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The path '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Path to the build configuration file.
    build_file_path = dir_path / DefaultValues.BUILD_FILE_NAME
    docker_image_name = read_build_configuration(build_file_path)['DOCKER_IMAGE_NAME']

    # Path to the configuration file.
    run_configuration_file_path = dir_path / DefaultValues.RUN_CONFIGURATION_FILE_NAME
    run_configuration = parse_run_configuration_file(run_configuration_file_path)

    print()

    mount_paths = run_configuration["mount_paths"]

    # Run command
    # -it - i stands for interactive, so this flag makes sure that standard
    # input ('STDIN') remains open even if you're not attached to container.
    # -t stands for pseudo-TTY, allocates a pseudo terminal inside container,
    # used to make environment inside container feel like a regular shell
    # session.
    docker_run_command = \
        f"docker run -v {get_project_directory()}:/InServiceOfX --gpus all -it "

    # Add mount paths from configuration file
    for mount_path in mount_paths:
        docker_run_command += f"-v {mount_path} "

    # -e flag sets environment and enables CUDA Forward Compatibility instead of
    # default CUDA Minor Version Compatibility.
    docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

    # Add the port 7860 for gradio applications.
    docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host "
    docker_run_command += docker_image_name

    print(docker_run_command)

    os.system(docker_run_command)

if __name__ == "__main__":
    main()