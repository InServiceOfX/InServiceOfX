#!/bin/bash

# Global variables
DOCKER_IMAGE_NAME="from-nvidia-python-24.02"

# If NVIDIA's RAFT can be installed on the current platform, then do so.
DISABLE_RAFT="OFF"

# Default value is 75;72. `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying
# which GPU architectures to build against.
CUDA_ARCHITECTURES="75;72"

function print_help
{
  echo "Usage: $0 [-h|--help] [--disable-raft]"
  echo ""
  echo "Options:"
  echo "-h, --help               Print this help message."
  echo "-r, --disable-raft       Disable RAFT implementation"
  exit 1
}

build_docker_image()
{
  echo "This is DISABLE_RAFT option: $DISABLE_RAFT"
  echo "This is CUDA_ARCHITECTURES option: $CUDA_ARCHITECTURES"

  # Builds from Dockerfile in this directory.
  # https://docs.docker.com/engine/reference/commandline/build/
  # --no-cache Don't use cache when building image.
  # Otherwise when we do run
  #  => [2/2] RUN pip install neuraloperators                                  4.0s
  # it would be cached without the option flag, not guaranteeing it was installed.
  docker build --build-arg="DISABLE_RAFT=${DISABLE_RAFT}" \
    --build-arg="COMPUTE_CAPABILITY=${CUDA_ARCHITECTURES}" \
    -t "$DOCKER_IMAGE_NAME" .
}

main()
{
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --disable-raft)
        DISABLE_RAFT="ON"
        shift # past argument
        ;;
      --help)
        print_help
        exit 0
        ;;
      *)
        # Unknown option
        print_help
        exit 1
        ;;
    esac
  done

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

  build_docker_image
}

main "$@"
