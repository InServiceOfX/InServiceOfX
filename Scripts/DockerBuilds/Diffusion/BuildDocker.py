#!/usr/bin/env python3

import argparse
import sys
import subprocess
from pathlib import Path

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from CommonUtilities import (
    DefaultValues,
    run_command,
    concatenate_dockerfiles)

from Utilities import (
    ReadBuildConfigurationWithNVIDIAGPU)

def print_help():
    help_text = """
Usage: build_docker_image.py [--no-cache]

Options:
  --no-cache         If provided, the Docker build will be performed without using cache
  --help             Show this help message and exit
"""
    print(help_text)

def build_docker_image(
    dockerfile_path,
    build_configuration,
    use_cache,
    build_context):
    """
    Builds the Docker image using the provided Dockerfile and build arguments.

    Args:
        dockerfile_path (Path): Path to the Dockerfile.
        build_configuration: Typically result from read_build_configuration.
        docker_image_name (str): Name of the Docker image to build.
        use_cache (bool): Whether to use Docker cache during build.
        build_context (Path): The directory to use as the build context.

    Raises:
        subprocess.CalledProcessError: If the Docker build command fails.
    """
    # See https://docs.docker.com/build/buildkit/
    docker_build_cmd = ["DOCKER_BUILDKIT=1", "docker", "build"]

    if not use_cache:
        docker_build_cmd.append("--no-cache")

    build_argument_keys = ["ARCH", "PTX", "COMPUTE_CAPABILITY"]

    # Add build arguments
    for key in build_argument_keys:
        docker_build_cmd.extend([
            "--build-arg",
            f"{key}={build_configuration[key]}"])

    # Specify Dockerfile
    docker_build_cmd.extend(["-f", str(dockerfile_path)])

    # Tag the image
    docker_build_cmd.extend([
        "-t",
        build_configuration['DOCKER_IMAGE_NAME']])

    # Specify build context
    docker_build_cmd.append(".")

    # Convert command list to string
    command_str = ' '.join(docker_build_cmd)

    run_command(command_str, cwd=build_context)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Build Docker image for Image and Video diffusion models.",
        add_help=False)
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='If provided, the Docker build will be performed without using cache')
    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help message and exit')

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    # Determine script and parent directories
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    # Where common files for building and running Docker images are stored.
    parent_dir = script_dir.parent

    # Path to build_configuration.txt
    build_configuration_path = script_dir / DefaultValues.BUILD_FILE_NAME

    try:
        configuration = ReadBuildConfigurationWithNVIDIAGPU().read_build_configuration(
            build_configuration_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Path to Dockerfile in script directory
    dockerfile_path = script_dir / "Dockerfile"

    # Paths to Dockerfile components
    dockerfile_header = parent_dir / "CommonFiles" / "Dockerfile.header"
    dockerfile_base = script_dir / "Dockerfile.base"
    dockerfile_huggingface = script_dir / "Dockerfile.huggingface"
    dockerfile_third_parties = script_dir / "Dockerfile.third_parties"

    try:
        concatenate_dockerfiles(
            dockerfile_path,
            dockerfile_header,
            dockerfile_base,
            dockerfile_huggingface,
            dockerfile_third_parties,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure Dockerfile exists
    if not dockerfile_path.is_file():
        print(
            f"Error: Dockerfile '{dockerfile_path}' does not exist.",
            file=sys.stderr)
        sys.exit(1)

    # Build the Docker image
    build_docker_image(
        dockerfile_path=dockerfile_path,
        build_configuration=configuration,
        use_cache=not args.no_cache,
        build_context=parent_dir
    )

    print(
        f"Successfully built Docker image '{configuration['DOCKER_IMAGE_NAME']}'.")

if __name__ == "__main__":
    main()
