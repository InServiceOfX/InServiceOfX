#!/bin/bash

# Global variables
DOCKER_IMAGE_NAME="langchain-local-nvidia-python-24.01"

print_help()
{
  echo "Usage: $0 [--enable-faiss]"
  echo
  echo "Options:"
  echo " --enable-faiss    If provided, the Docker build includes the installation of FAISS."
  echo "                    By default, FAISS is not installed."
  echo " --no-cache         If provided, the Docker build will be performed without using cache"
  echo
  echo "Example:"
  echo "  $0 --enable-faiss     # Builds Docker image with FAISS installation."
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

# Function to run a command and check the exit code
run_command_and_check_exit_code() {
    local command="$1"
    local command_output

    # Run the command and capture the output
    command_output=$(eval "$command" 2>&1)
    command_exit_code=$?

    # Check if the command executed successfully
    if [ $command_exit_code -eq 0 ]; then
        echo "$command_output"
    else
        # Check if the error is due to permission denied
        if [ $command_exit_code -eq 126 ]; then
            echo "Error: Permission denied. Cannot run the command: $command"
            return $command_exit_code
        else
            echo "Error: Command '$command' failed with exit code $command_exit_code"
            echo "$command_output"
            return $command_exit_code
        fi
    fi
}

main()
{
  echo "Con(cat)enating Dockerfiles: "

  command_cat_dockerfiles="cat Dockerfile.header \
    Dockerfile.base \
    Dockerfile.langchain \
    Dockerfile.huggingface \
    Dockerfile.singleThirdParties \
    Dockerfile.llama_index \
    > Dockerfile"

  run_command_and_check_exit_code "$command_cat_dockerfiles"

  # Check if the command executed successfully
  if [ $command_exit_code -eq 0 ]; then
      echo "Con(cat)enating Dockerfiles executed successfully."
  else
      # Check if the error is due to permission denied
      if [ $command_exit_code -eq 126 ]; then
          echo "Error: Permission denied. Cannot run the command."
          exit $command_exit_code
      else
          echo "Error: Command failed with exit code $command_exit_code."
          exit $command_exit_code
      fi
  fi

  build_docker_image "$@"
}

main "$@"
