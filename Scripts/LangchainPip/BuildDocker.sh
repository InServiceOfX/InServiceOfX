#!/bin/bash

# You must run this in the directory the this script is in because it needs to
# find the Dockerfile.

build_docker_image()
{
  # Builds from Dockerfile in this directory.
  # https://docs.docker.com/engine/reference/commandline/build/
  # --no-cache Don't use cache when building image.
  # Otherwise when we do run
  #  => [2/2] RUN pip install neuraloperators                                  4.0s
  # it would be cached without the option flag, not guaranteeing it was installed.
  docker build -t "$DOCKER_IMAGE_NAME" .
}

main()
{
  build_docker_image
}

main "$@"
