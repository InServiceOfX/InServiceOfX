#!/bin/bash
# An example for the user input would be
# /home/propdev/Prop/InServiceOfX:/InServiceOfX

# Global variables
DOCKER_IMAGE_NAME="from-nvidia-python-24.02"

get_base_directory()
{
  # $0 gives script's name along with its relative path.
  script_directory="$(dirname "$(realpath "$0")")"
  base_directory="${script_directory}/../../"

  if [ -d "$base_directory" ]; then
    echo "$base_directory"
  else
    echo "$base_directory is not an existing directory"
    exit 1
  fi
}

get_mount_path()
{
  local base_directory=$(get_base_directory)
  local mount_path="$base_directory/"
  echo "$mount_path"
}

# Check if there is any user input
if [ -z "$1" ]; then

  # No user input, so default to an expected mount path.

  path_to_mount=$(get_mount_path)
  echo "Mount default path: $path_to_mount"

else

  # User input provided, so that the first argument as path
  path_to_mount="$1"
  echo "Mount path: $path_to_mount"
fi

# Check if path is an existing path
# [ is start of a test command condition. -d is a test for a directory.
if [ ! -d "$path_to_mount" ]; then
  echo "The path '$path_to_mount' is not an existing path."
  exit 1
fi

# Run command
# -it - i stands for interactive, so this flag makes sure that standard input
# ('STDIN') remains open even if you're not attached to container.
# -t stands for pseudo-TTY, allocates a pseudo terminal inside container, used
# to make environment inside container feel like a regular shell session.
command="docker run -v $path_to_mount:/InServiceOfX --gpus all -it "
# -e flag sets environment and enables CUDA Forward Compatibility instead of
# default CUDA Minor Version Compatibility.
command+="-e NVIDIA_DISABLE_REQUIRE=1 "
command+="-p 8888:8888 --rm --ipc=host --ulimit memlock=-1 --ulimit "
# Originally it was this command to run the base image, but we've added layers
# so that we 
# command+="stack=67108864 nvcr.io/nvidia/pytorch:24.02-py3 "
command+="stack=67108864 $DOCKER_IMAGE_NAME "

echo $command
# eval is part of POSIX and executes arguments like a shell command.
eval $command