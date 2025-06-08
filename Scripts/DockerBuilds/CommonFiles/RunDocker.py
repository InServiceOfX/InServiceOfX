"""
@brief Similar to bash script RunDocker.sh, but the Python version allows for
more options.

@details
USAGE: python ./RunDocker.py [directory_path] [--gpu GPU_ID]
"""

from pathlib import Path
import os, sys
import argparse
import subprocess
import yaml

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from CommonUtilities import (
    DefaultValues,
    get_docker_builds_directory,
    get_project_directory,
    parse_run_configuration_file)

from Utilities import (
    CreateDockerRunCommand,
    ReadBuildConfigurationForMinimalStack)


def print_help():
    help_text = """
Usage: RunDocker.py [directory_path] [--gpu GPU_ID]

Options:
  directory_path      Path to the directory containing build configuration files
  --gpu GPU_ID       Specific GPU ID to use (non-negative integer). If not specified, uses all GPUs.
  --help             Show this help message and exit
"""
    print(help_text)


def validate_gpu_id(value):
    try:
        gpu_id = int(value)
        if gpu_id < 0:
            raise ValueError("GPU ID must be non-negative")
        return gpu_id
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


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
        '--gpu',
        type=validate_gpu_id,
        help='Specific GPU ID to use (non-negative integer)')
    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help message and exit')

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

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
    build_configuration = ReadBuildConfigurationForMinimalStack().read_build_configuration(
        build_file_path)

    # Path to the configuration file.
    run_configuration_file_path = dir_path / DefaultValues.RUN_CONFIGURATION_FILE_NAME
    run_configuration = parse_run_configuration_file(run_configuration_file_path)

    print()

    # Check for docker-compose.yml first
    databases_dir = dir_path / "Databases"
    docker_compose_path = databases_dir / "docker-compose.yml"

    network_name = None
    if docker_compose_path.exists():
        try:
            # Run docker-compose up -d
            subprocess.run(
                [
                    'docker',
                    'compose',
                    '-f',
                    str(docker_compose_path),
                    'up',
                    '-d'],
                check=True
            )

            # Get network name from docker-compose.yml
            with open(docker_compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)
                if 'networks' in compose_config:
                    network_name = next(iter(compose_config['networks'].keys()))
        except Exception as e:
            print(f"Warning: Failed to start docker-compose: {e}")

    # Add network to run configuration if we have one
    if network_name:
        run_configuration['network'] = network_name

    # Create and run docker command
    create_docker_run_command = CreateDockerRunCommand(
        get_project_directory(),
        build_configuration,
        run_configuration,
        gpu_id=args.gpu)

    os.system(create_docker_run_command.docker_run_command)

if __name__ == "__main__":
    main()
