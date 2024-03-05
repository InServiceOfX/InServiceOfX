#!/bin/bash

# Global variables
DOCKER_IMAGE_NAME="from-nvidia-python-24.02"

# If NVIDIA's RAFT can be installed on the current platform, then do so.
DISABLE_RAFT="OFF"

# Default value is 75;72. `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying
# which GPU architectures to build against.
CUDA_ARCHITECTURES="75;72"

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
    esac
  done

  cat Dockerfile.header \
    Dockerfile.base \
    Dockerfile.singleThirdParties \
    Dockerfile.huggingface \
    Dockerfile.langchain \
    > Dockerfile

  # Get CUDA Architecture.
  source ../GetComputeCapability.sh
  CUDA_ARCHITECTURES=$(get_compute_capability_as_cuda_architecture)  

  echo "CUDA Compute capability determined to be: $CUDA_ARCHITECTURES"

  build_docker_image
}

main "$@"
