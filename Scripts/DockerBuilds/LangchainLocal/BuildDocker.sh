#!/bin/bash

# Global variables
DOCKER_IMAGE_NAME="langchain-local-nvidia-python-24.01"

print_help()
{
  echo "Usage: $0 [--enable-faiss]"
  echo
  echo "Options:"
  echo " --eenable-faiss    If provided, the Docker build includes the installation of FAISS."
  echo "                    By default, FAISS is not installed."
  echo
  echo "Example:"
  echo "  $0 --eanble-faiss     # Builds Docker image with FAISS installation."
  echo "  $0                    # Builds Docker image without FAISS installation." 
}

# Reads ARCH and PTX values from local text file, typically in the same
# (sub)directory as this script and named nvidia_compute_capabilities.txt.
read_compute_capabilities()
{
  local file_path="$1"
  local arch_value=$(grep '^ARCH=' "$file_path" | cut -d '=' -f 2)
  local ptx_value=$(grep '^PTX=' "$file_path" | cut -d '=' -f 2)
  local compute_capability_value=$(grep '^COMPUTE_CAPABILITY=' "$file_path" | cut -d '=' -f 2)

  echo "$arch_value" "$ptx_value" "$compute_capability_value"
}

# You must run this in the directory the this script is in because it needs to
# find the Dockerfile.

build_docker_image()
{
  # Check for help option
  for arg in "$@"
  do
    if [ "$arg" = "--help" ]; then
      print_help
      exit 0
    fi
  done

  # Path to nvidia_compute_capabilities.txt file; the hard assumption is made
  # that it'll be in the exact same (sub)directory as this file.
  local capabilities_file="$(dirname "$0")/nvidia_compute_capabilities.txt"

  # Read ARCH and PTX values.
  read -r ARCH_VALUE PTX_VALUE COMPUTE_CAPABILITY < <(read_compute_capabilities \
    "$capabilities_file")

  echo "Current directory: $(pwd)"
  cd ../ || { echo "Failed to change directory to '../'"; exit 1; }

  local enable_faiss=false

  # Check for --enalbe-false flag
  for arg in "$@"
  do 
    if [ "$arg" = "--enable-faiss" ]; then
      enable_faiss=true
      break
    fi
  done

  # Builds from Dockerfile in this directory.
  docker build \
    --build-arg ARCH="$ARCH_VALUE" \
    --build-arg PTX="$PTX_VALUE" \
    --build-arg COMPUTE_CAPABILITY="$COMPUTE_CAPABILITY" \
    -t "$DOCKER_IMAGE_NAME" \
    -f LangchainLocal/Dockerfile .
}

main()
{
  cat Dockerfile.header \
    Dockerfile.base \
    Dockerfile.singleThirdParties \
    Dockerfile.langchain \
    > Dockerfile

  build_docker_image "$@"
}

main "$@"
