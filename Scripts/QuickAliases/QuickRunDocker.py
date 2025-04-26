#!/usr/bin/env python3
"""
QuickRunDocker.py - Quickly run Docker containers with predefined
configurations.

This script reads configuration files to set up Docker run commands with the
correct volumes and image name. It supports specifying GPU devices and custom
build directories.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import os

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# The main project directory that will be mounted in the Docker container
# MODIFY THIS to match your environment's project directory path
PROJECT_DIR = "/home/propdev/Prop/InServiceOfX"

# The path where the project will be mounted inside the Docker container
# Usually this doesn't need to be changed
CONTAINER_MOUNT_PATH = "/InServiceOfX"

# ==============================================================================


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=\
            "Quickly run Docker containers with predefined configurations."
    )
    parser.add_argument(
        "--build-dir", 
        type=str,
        default="../DockerBuilds/Builds/LLM/LocalLLMFull/",
        help="Path to directory containing Docker configuration files"
    )
    parser.add_argument(
        "--gpu", 
        type=int,
        help="GPU device number to use"
    )
    parser.add_argument(
        "--port", 
        type=int,
        nargs="+",
        default=[8888, 7860],
        help="Ports to expose (default: 8888 7860)"
    )
    
    return parser.parse_args()


def get_script_dir():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def resolve_build_dir(build_dir_path):
    """Resolve the build directory path, handling relative paths."""
    build_dir = Path(build_dir_path)
    
    # If it's a relative path, make it relative to the script directory
    if not build_dir.is_absolute():
        build_dir = get_script_dir() / build_dir
    
    # Resolve to absolute path
    build_dir = build_dir.resolve()
    
    if not build_dir.exists():
        print(f"Error: Build directory '{build_dir}' does not exist.")
        sys.exit(1)
        
    return build_dir


def read_docker_image_name(build_dir):
    """Read the Docker image name from build_docker_configuration.txt."""
    config_file = build_dir / "build_docker_configuration.txt"
    
    if not config_file.exists():
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        for line in f:
            if line.startswith("DOCKER_IMAGE_NAME="):
                return line.strip().split("=", 1)[1]
    
    print(f"Error: DOCKER_IMAGE_NAME not found in '{config_file}'.")
    sys.exit(1)


def read_mount_volumes(build_dir):
    """Read mount volumes from run_docker_configuration.txt."""
    config_file = build_dir / "run_docker_configuration.txt"
    
    if not config_file.exists():
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    
    volumes = []
    print("\nReading volume mounts:")
    print("-" * 50)
    
    with open(config_file, 'r') as f:
        for line in f:
            if line.startswith("MOUNT_PATH_"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    volume_path = parts[1]
                    volumes.append(volume_path)
                    
                    # Print the volume mount with source and destination
                    if ":" in volume_path:
                        source, destination = volume_path.split(":", 1)
                        print(f"Source:      {source}")
                        print(f"Destination: {destination}")
                    else:
                        print(f"Volume:      {volume_path}")
                    print("-" * 50)
    
    if not volumes:
        print("No volume mounts found in configuration file.")
    
    return volumes


def build_docker_command(image_name, volumes, gpu_device, ports):
    """Build the Docker run command."""
    cmd = ["docker", "run"]
    
    # Add volume mounts
    cmd.extend(["-v", f"{PROJECT_DIR}:{CONTAINER_MOUNT_PATH}"])
    
    for volume in volumes:
        cmd.extend(["-v", volume])
    
    # Add GPU configuration if specified
    if gpu_device is not None:
        cmd.extend(["--gpus", f"device={gpu_device}"])
    
    # Add interactive terminal
    cmd.append("-it")
    
    # Add environment variables
    cmd.extend(["-e", "NVIDIA_DISABLE_REQUIRE=1"])
    
    # Add port mappings
    for port in ports:
        cmd.extend(["-p", f"{port}:{port}"])
    
    # Add remaining options
    cmd.extend(["--rm", "--ipc=host", image_name])
    
    return cmd


def run_docker_with_app(docker_cmd):
    docker_cmd.extend([
        "bash",
        "-c",
        "cd /ThirdParty/InServiceOfX/PythonApplications/CLIChatLocal/Executables && python main_CLIChatLocal.py --dev"
    ])
    
    try:
        # If container already exists, remove it
        subprocess.run(docker_cmd)
    except:
        pass
        
    # Run the Docker command
    subprocess.run(docker_cmd)


def main():
    """Main function to run the Docker container."""
    args = parse_arguments()
    
    # Resolve build directory
    build_dir = resolve_build_dir(args.build_dir)
    print(f"Using build directory: {build_dir}")
    
    # Read Docker image name
    image_name = read_docker_image_name(build_dir)
    print(f"Using Docker image: {image_name}")
    
    # Read mount volumes
    volumes = read_mount_volumes(build_dir)
    print(f"Found {len(volumes)} volume mounts")
    
    # Build and execute Docker command
    docker_cmd = build_docker_command(image_name, volumes, args.gpu, args.port)
    
    print("\nExecuting Docker command:")
    print(" ".join(docker_cmd))
    print("\n")
    
    # Execute the command
    run_docker_with_app(docker_cmd)


if __name__ == "__main__":
    main()
