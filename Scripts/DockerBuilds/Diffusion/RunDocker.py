"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py
"""

from pathlib import Path
import os, sys

# Global variables
DOCKER_IMAGE_NAME="diffusion-nvidia-python-24.01"
CONFIGURATION_FILE_NAME="mount_paths.txt"

def get_project_directory():
    # Resolve to absolute path.
    current_filepath = Path(__file__).resolve()

    # This variable's value depends on the location of this file relative to
    # other subdirectories.
    parent_subdirectories = 3

    project_directory = current_filepath.parents[parent_subdirectories]

    if not project_directory.is_dir():
        print(f"{project_directory} is not a directory")
        exit(1)

    if not project_directory.exists():
        print(f"{project_directory} is not an existing directory")
        exit(1)

    return project_directory

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

                if ':' in stripped_line and Path(line.split(':', 1)[0]).resolve().is_dir():
                    mount_paths.append(line)
                else:
                    print(f"Invalid or nonexistent path in file: {line.split(':', 1)[0]}")
    return mount_paths

def main():
    project_directory = get_project_directory()

    # Run command
    # -it - i stands for interactive, so this flag makes sure that standard
    # input ('STDIN') remains open even if you're not attached to container.
    # -t stands for pseudo-TTY, allocates a pseudo terminal inside container,
    # used to make environment inside container feel like a regular shell
    # session.
    docker_run_command = f"docker run -v {project_directory}:/InServiceOfX --gpus all -it "

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

    for mount_path in parse_mount_paths_file():
        docker_run_command += f"-v {mount_path} "

    # -e flag sets environment and enables CUDA Forward Compatibility instead of
    # default CUDA Minor Version Compatibility.
    docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

    # Add the port 7860 for gradio applications.
    docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host --ulimit memlock=-1 "

    docker_run_command += "--ulimit stack=67108864 "
    docker_run_command += DOCKER_IMAGE_NAME

    print(docker_run_command)

    os.system(docker_run_command)


if __name__ == "__main__":
    main()