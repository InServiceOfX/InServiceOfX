#!/bin/bash

# Source the ParseBuildConfiguration.sh file
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PARENT_DIR/CommonUtilities.sh"

print_help()
{
  echo "Usage: $0 [--no-cache]"
  echo
  echo "Options:"
  echo " --no-cache         If provided, the Docker build will be performed without using cache"
  echo
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

  # Path to build_docker_configuration.txt file; the hard assumption is made
  # that it'll be in the exact same (sub)directory as this file.
  local configuration_file="$(dirname "$0")/build_docker_configuration.txt"

  parse_build_configuration_file "$configuration_file"

  # Check if DOCKER_IMAGE_NAME is set.
  if [ -z "$DOCKER_IMAGE_NAME" ]; then
    echo "Error: DOCKER_IMAGE_NAME is not set. Please check your configuration file."
    exit 1
  fi

  echo "$use_cache"
  echo "$DOCKER_IMAGE_NAME"
  echo "$script_dir"

  # Builds from Dockerfile in this directory.
  docker build $use_cache \
    -t "$DOCKER_IMAGE_NAME" \
    -f "$script_dir/Dockerfile" .
}


main()
{
  echo "Con(cat)enating Dockerfiles: "

  local dockerfile_header="$PARENT_DIR/CommonFiles/Dockerfile.header"

  command_cat_dockerfiles="cat $dockerfile_header \
    Dockerfile.base \
    Dockerfile.nodejs \
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
