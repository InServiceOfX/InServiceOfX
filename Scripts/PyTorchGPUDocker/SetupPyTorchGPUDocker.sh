#!/bin/bash

# You must run this in the directory the this script is in because it needs to
# find the Dockerfile.

# Global variables
DOCKER_IMAGE_NAME="from-nvidia-python-24.02"
# As of March 2, 2024
NVIDIA_PYTORCH_TAG="nvcr.io/nvidia/pytorch:24.02-py3"

# If NVIDIA's RAFT can be installed on the current platform, then do so.
DISABLE_RAFT="OFF"

# Default value is 75;72. `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying
# which GPU architectures to build against.
CUDA_ARCHITECTURES="75;72"

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
  local mount_path="$base_directory/:/InServiceOfX"
  echo "$mount_path"
}

install_docker_compose_plugin()
{
  # For docker-compose
  sudo apt-get update
  sudo apt-get install docker-compose-plugin
}

pull_and_build_docker_image()
{
  # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
  # Also, see
  # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
  # Check for the latest version on this page, on the left:
  # https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
  # Look for the PyTorch Release Notes.
  # Also, try
  # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
  # and click on "Tags" on the top center for the different tags.

  docker pull "$NVIDIA_PYTORCH_TAG"

  # Builds from Dockerfile in this directory.
  # https://docs.docker.com/engine/reference/commandline/build/
  # --no-cache Don't use cache when building image.
  # Otherwise when we do run
  #  => [2/2] RUN pip install neuraloperators                                  4.0s
  # it would be cached without the option flag, not guaranteeing it was installed.
  docker build --no-cache --build-arg="DISABLE_RAFT=${DISABLE_RAFT}" \
    --build-arg="COMPUTE_CAPABILITY=${CUDA_ARCHITECTURES}" \
    -t "$DOCKER_IMAGE_NAME" .
}

optional_run_commands()
{
  # --rm Removes the container, meant for short-lived process, perform specific task,
  # upon exit. You ensure temporary containers don't accumulate on system. --rm only
  # removes container instance not the image.
  docker run -v /home/propdev/Prop/InServiceOfX:/InServiceOfX --gpus all -it --rm "$NVIDIA_PYTORCH_TAG"

  # -p 8888:8888 maps port 8888 inside container to port 8888 on host machine.

  # When running this Docker, it said SHMEM (Shared memory?) is set too low, and so it suggested
  # these options: --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

  docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
}

run_docker_container()
{
  local mount_path=$(get_mount_path)

  echo "This is the mount path: $mount_path"

  docker run -v "$mount_path" --gpus all -e NVIDIA_DISABLE_REQUIRE=1 -it -p \
    8888:8888 --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    "$DOCKER_IMAGE_NAME"
}

main()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --disable-raft)
        DISABLE_RAFT="ON"
        shift # past argument
        ;;
    esac
  done

  install_docker_compose_plugin

  # TODO: at this point, consider combining Dockerfile fragments
  # cat Dockerfile.base Dockerfile.faiss Dockerfile.langchain > Dockerfile
  # To keep Dockerfile modular.

  cat Dockerfile.header \
    Dockerfile.base \
    Dockerfile.langchain \
    Dockerfile.huggingface \
    Dockerfile.singleThirdParties \
    > Dockerfile

  # Get CUDA Architecture.
  source ../GetComputeCapability.sh
  CUDA_ARCHITECTURES=$(get_compute_capability_as_cuda_architecture)

  echo "CUDA Compute capability determined to be: $CUDA_ARCHITECTURES"

  pull_and_build_docker_image
  run_docker_container
}

main "$@"

# You may see the following if you run 
#jupyter notebook
# in the command prompt:
#     Or copy and paste this URL:
#        http://hostname:8888/?token=cf73dc6455ca527875631fde4f24511067f751e3478c5482
# You want to keep the token. But hostname you may have to replace with localhost, and so this
# works in your browser:
#        http://localhost:8888/?token=cf73dc6455ca527875631fde4f24511067f751e3478c5482
