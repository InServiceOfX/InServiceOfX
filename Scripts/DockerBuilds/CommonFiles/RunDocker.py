"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py [directory_path] [--arm64]
"""

from pathlib import Path
import os, sys
import argparse

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from CommonUtilities import (
    CreateDockerRunCommand,
    DefaultValues,
    get_docker_builds_directory,
    get_project_directory,
    parse_run_configuration_file,
    read_build_configuration)


def print_help():
    help_text = """
Usage: RunDocker.py [directory_path] [--arm64]

Options:
  directory_path      Path to the directory containing build configuration files
  --arm64             Run for ARM 64 architecture
  --help              Show this help message and exit
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="Run Docker container.",
        add_help=False)

    parser.add_argument(
        'directory_path',
        nargs='?',
        default=None,
        help='Path to the directory containing build configuration files')
    parser.add_argument(
        '--arm64',
        action='store_true',
        help='Run for ARM 64 architecture')
    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help message and exit')

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    is_arm64 = args.arm64
    docker_builds_directory = get_docker_builds_directory()

    if args.directory_path:
        dir_path = Path(args.directory_path).resolve()
    else:
        # Default to the specific directory structure
        dir_path = docker_builds_directory / "LLM" / "Meta" / "FullLlama"

    # Validate the directory path
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The path '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Path to the build configuration file.
    build_file_path = dir_path / DefaultValues.BUILD_FILE_NAME
    build_configuration = read_build_configuration(build_file_path)

    # Path to the configuration file.
    run_configuration_file_path = dir_path / DefaultValues.RUN_CONFIGURATION_FILE_NAME
    run_configuration = parse_run_configuration_file(run_configuration_file_path)

    print()

    create_docker_run_command = CreateDockerRunCommand(
        get_project_directory(),
        build_configuration,
        run_configuration,
        is_arm64)

    os.system(create_docker_run_command.docker_run_command)

if __name__ == "__main__":
    main()
