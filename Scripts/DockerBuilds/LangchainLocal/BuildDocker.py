#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[1]))
from CommonUtilities import (
    build_docker_image,
    read_build_configuration,
    DefaultValues,
    concatenate_dockerfiles)

def print_help():
    help_text = """
Usage: build_docker_image.py [--no-cache] [--arm64]

Options:
  --no-cache         If provided, the Docker build will be performed without using cache
  --help             Show this help message and exit
  --enable-faiss     If provided, the Docker build includes the installation of FAISS.
                     By default, FAISS is not installed.
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
        '--enable-faiss',
        action='store_true',
        help='Install FAISS from source')
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
    parent_dir = script_dir.parents[2]

    # Path to build_configuration.txt
    build_configuration_path = script_dir / DefaultValues.BUILD_FILE_NAME

    try:
        configuration = read_build_configuration(build_configuration_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Path to Dockerfile in script directory
    dockerfile_path = script_dir / "Dockerfile"

    # Paths to Dockerfile components
    dockerfile_header = parent_dir / "CommonFiles" / "Dockerfile.header"
    dockerfile_base = parent_dir / "CommonFiles" / "Dockerfile.base"
    dockerfile_rust = parent_dir / "CommonFiles" / "Dockerfile.rust"
    dockerfile_more_pip_installs = script_dir / "Dockerfile.more_pip_installs"
    dockerfile_huggingface = parent_dir / "CommonFiles" / "Dockerfile.huggingface"
    dockerfile_meta_llama = script_dir / "Dockerfile.meta-llama"
    dockerfile_apis = script_dir / "Dockerfile.apis"
    dockerfile_math = script_dir / "Dockerfile.math"
    dockerfile_third_parties = script_dir / "Dockerfile.third_parties"

    try:
        concatenate_dockerfiles(
            dockerfile_path,
            dockerfile_header,
            dockerfile_base,
            dockerfile_rust,
            dockerfile_more_pip_installs,
            dockerfile_huggingface,
            dockerfile_meta_llama,
            dockerfile_third_parties,
            dockerfile_apis,
            dockerfile_math,
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
        build_context=parent_dir)

    print(
        f"Successfully built Docker image '{configuration['DOCKER_IMAGE_NAME']}'.")

if __name__ == "__main__":
    main()


# You must run this in the directory the this script is in because it needs to
# find the Dockerfile.

build_docker_image()
{
  local enable_faiss=false
  local use_cache=""

  # Check for help option
  for arg in "$@"
  do
    if [ "$arg" = "--help" ]; then
      print_help
      exit 0
    elif [[ "$arg" == "--enable-faiss" ]]; then
      enable_faiss=true
    elif [[ "$arg" == "--no-cache" ]]; then
      use_cache="--no-cache"
    fi
  done

  # Determine the script's directory and ensure Dockerfile is there.
  local script_dir="$(dirname "$(realpath "$0")")"
  if [[ ! -f "$script_dir/Dockerfile" ]]; then
    echo "Dockerfile not found in script directory ($script_dir)."
    exit 1
  fi

  # Path to nvidia_compute_capabilities.txt file; the hard assumption is made
  # that it'll be in the exact same (sub)directory as this file.
  local capabilities_file="$(dirname "$0")/nvidia_compute_capabilities.txt"

  # Read ARCH and PTX values.
  read -r ARCH_VALUE PTX_VALUE COMPUTE_CAPABILITY < <(read_compute_capabilities \
    "$capabilities_file")

  # Go to parent directory because we want to use the BuildOpenCVWithCUDA.sh
  # script.
  echo "Current directory: $(pwd)"
  cd ../ || { echo "Failed to change directory to '../'"; exit 1; }

  # Construct build-args with optional FAISS.
  local build_args="--build-arg ARCH=$ARCH_VALUE --build-arg PTX=$PTX_VALUE "
  build_args+="--build-arg COMPUTE_CAPABILITY "

  if $enable_faiss; then
    build_args+="--build-arg ENABLE_FAISS=true "
  fi

  echo "$use_cache"
  echo "$build_args"
  echo "$DOCKER_IMAGE_NAME"
  echo "$script_dir"

  # Builds from Dockerfile in this directory.
  docker build $use_cache \
    $build_args \
    -t "$DOCKER_IMAGE_NAME" \
    -f "$script_dir/Dockerfile" .
}
