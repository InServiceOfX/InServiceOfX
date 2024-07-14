#!/bin/bash

# Global variables
DOCKER_IMAGE_NAME="minimal-cuda-nvidia-python-24.01"

print_help()
{
  echo "Usage: $0 [--no-cache]"
  echo
  echo "Options:"
  echo " --no-cache         If provided, the Docker build will be performed without using cache"
  echo
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
  local use_cache=""

  # Check for help option
  for arg in "$@"
  do
    if [ "$arg" = "--help" ]; then
      print_help
      exit 0
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

  echo "$use_cache"
  echo "$DOCKER_IMAGE_NAME"
  echo "$script_dir"

  # Builds from Dockerfile in this directory.
  docker build $use_cache \
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

  local script_dir="$(dirname "$(realpath "$0")")"
  local parent_dir="$(dirname "$script_dir")"

  local dockerfile_header="$parent_dir/CommonFiles/Dockerfile.header"

  command_cat_dockerfiles="cat $dockerfile_header \
    Dockerfile.base \
    Dockerfile.MoreNvidia \
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
