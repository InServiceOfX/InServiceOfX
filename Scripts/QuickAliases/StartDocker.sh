#!/bin/bash

# Make this an alias by
# echo "alias StartDocker='./StartDocker.sh'" >> ~/.bashrc
# source ~/.bashrc
# I found that I also needed to do this:
# chmod +x StartDocker.sh
# because otherwise trying to run it gives error of Permission denied.

# Set the base directory (modify this if needed)
BASE_DIR="/media/ernest/Samsung980ProPCI/PropD/InServiceOfX"

# Check if a base directory is provided as the first argument
if [ -n "$1" ]; then
    BASE_DIR="$1"
fi

# Set the argument for RunDocker.py (default value)
DOCKERFILE_LOCATION_ARG="../LLM/Meta/FullLlama/"

# Check if a Docker argument is provided as the second argument
if [ -n "$2" ]; then
    DOCKERFILE_LOCATION_ARG="$2"
fi

# Navigate to the base directory
cd "$BASE_DIR" || {
    echo "Directory $BASE_DIR not found. Exiting."
    exit 1
}

# Activate the virtual environment
source ./venv/bin/activate

# Navigate to Scripts/PyTorchGPUDocker
cd Scripts/DockerBuilds/CommonFiles || {
    echo "Directory Scripts/DockerBuilds/CommonFiles not found. Exiting."
    exit 1
}

# Run the Python script with the specified argument
python RunDocker.py "$DOCKERFILE_LOCATION_ARG"
