#!/bin/bash

# Look it up here: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
NVIDIA_DOCKER_NAME="nvcr.io/nvidia/pytorch:"
# By doing nvcc --version, one can see that from 24.03, CUDA is 12.4. From
# 24.02, CUDA is 12.3.r12.3. For OpenCV and this issue, let's stay with < 12.4.
# Issue: https://github.com/opencv/opencv_contrib/issues/3690
VERSION="24.02"

# Define the function to get the parent's parent directory
get_parent_parent_dir()
{
  local current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  local parent_dir="$(dirname "$current_dir")"
  echo "$parent_dir"  
}

main()
{
  echo $(get_parent_parent_dir)

  local docker_options="--gpus all -it --rm "
  local volume_mapping="-v $(get_parent_parent_dir):/InServiceOfX/ "
  local port_mapping="-p 8888:8888 "
  local docker_image="$NVIDIA_DOCKER_NAME$VERSION-py3"

  local docker_command="docker run $docker_options $volume_mapping $port_mapping $docker_image"

  echo "$docker_command"
  $docker_command
}

main
