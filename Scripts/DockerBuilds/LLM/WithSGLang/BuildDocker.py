#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from CommonUtilities import (
    DefaultValues,
    concatenate_dockerfiles)

from Utilities import (
    BuildDockerImageWithNVIDIAGPU,
    ReadBuildConfigurationWithNVIDIAGPU)

def print_help():
    help_text = """
Usage: build_docker_image.py [--no-cache] [--arm64]

Options:
  --no-cache         If provided, the Docker build will be performed without using cache
  --help             Show this help message and exit
"""
    print(help_text)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Build Docker image for LLM models.",
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
    parent_dir = script_dir.parents[1]

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
    dockerfile_base = parent_dir / "CommonFiles" / "Dockerfile.base"
    dockerfile_rust = parent_dir / "CommonFiles" / "Dockerfile.rust"
    dockerfile_opencv_with_cuda = parent_dir / "CommonFiles" / \
        "Dockerfile.opencv_with_cuda"
    dockerfile_sglang = script_dir / "Dockerfile.sglang"

    try:
        concatenate_dockerfiles(
            dockerfile_path,
            dockerfile_header,
            dockerfile_base,
            dockerfile_opencv_with_cuda,
            dockerfile_rust,
            dockerfile_sglang,
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
    BuildDockerImageWithNVIDIAGPU().build_docker_image(
        dockerfile_path=dockerfile_path,
        build_configuration=configuration,
        use_cache=not args.no_cache,
        build_context=parent_dir)

    print(
        f"Successfully built Docker image '{configuration['DOCKER_IMAGE_NAME']}'.")

if __name__ == "__main__":
    main()
