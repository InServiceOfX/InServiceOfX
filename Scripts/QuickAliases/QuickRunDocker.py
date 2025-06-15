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
import yaml
from pathlib import Path

def read_project_dir_from_config():
    """Read PROJECT_DIR from configuration file."""
    config_file = Path(__file__).parent / "quick_run_docker_configuration.txt"

    if not config_file.exists():
        print("\nError: Configuration file not found!")
        print(f"Please create: {config_file}")
        print("\nThe file should contain a line like:")
        print('PROJECT_DIR = "/path/to/your/project"')
        print('or')
        print('PROJECT_DIR="/path/to/your/project"')
        print("\nThis should be the path to your main project directory that will be mounted in Docker.")
        sys.exit(1)

    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('PROJECT_DIR'):
                    # Handle both "=" and " = " formats
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        # Remove quotes and whitespace
                        project_dir = parts[1].strip().strip('"\'')
                        return project_dir
        
        print("\nError: PROJECT_DIR not found in configuration file!")
        print(f"Please add PROJECT_DIR to: {config_file}")
        print("\nFormat should be:")
        print('PROJECT_DIR = "/path/to/your/project"')
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError reading configuration file: {e}")
        sys.exit(1)

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Read the project directory from configuration file
PROJECT_DIR = read_project_dir_from_config()

# The path where the project will be mounted inside the Docker container
# Usually this doesn't need to be changed
CONTAINER_MOUNT_PATH = "/InServiceOfX"

# Base directory for Docker builds
DOCKER_BUILDS_BASE = "../DockerBuilds/Builds/"

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
        default="LLM/LocalLLMFull/",
        help=\
            "Path relative to ../DockerBuilds/Builds/ (e.g., LLM/LocalLLMFull/)"
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
    parser.add_argument(
        "--clichatlocal",
        action="store_true",
        help="Run the CLI chat local application instead of dropping to shell"
    )
    
    return parser.parse_args()


def get_script_dir():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def resolve_build_dir(build_dir_path):
    """Resolve the build directory path, handling relative paths."""
    # Combine the base directory with the provided path
    full_path = Path(DOCKER_BUILDS_BASE) / build_dir_path
    
    # If it's a relative path, make it relative to the script directory
    if not full_path.is_absolute():
        full_path = get_script_dir() / full_path
    
    # Resolve to absolute path
    full_path = full_path.resolve()
    
    if not full_path.exists():
        print(f"Error: Build directory '{full_path}' does not exist.")
        sys.exit(1)
        
    return full_path


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

def read_database_docker_compose(build_dir):
    """Only if there is a 'Databases' subdirectory with docker-compose.yml will
    this run docker-compose as a subprocess.
    """
    databases_dir = build_dir / "Databases"
    docker_compose_path = databases_dir / "docker-compose.yml"

    if not docker_compose_path.exists():
        return None

    try:
        # Run docker-compose up -d
        subprocess.run(
            [
                'docker',
                'compose',
                '-f',
                str(docker_compose_path),
                'up',
                '-d'],
            check=True
        )

        # Get network name from docker-compose.yml
        with open(docker_compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
            if 'networks' in compose_config:
                network_name = next(iter(compose_config['networks'].keys()))
        return network_name
    except Exception as e:
        print(f"Warning: Failed to start docker-compose: {e}")
        return None

def build_docker_command(
        image_name,
        volumes,
        gpu_device,
        ports,
        database_network = None):
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
    
    # Add network if specified
    if database_network is not None:
        cmd.extend(["--network", database_network])
    
    # Add remaining options
    cmd.extend(["--rm", "--ipc=host", image_name])
    
    return cmd


def run_docker_with_shell(docker_cmd):
    """Run Docker container and drop into an interactive shell."""
    docker_cmd.extend([
        "bash",
        "-c",
        "cd /InServiceOfX/PythonLibraries && bash"
    ])
    
    try:
        subprocess.run(docker_cmd)
    except KeyboardInterrupt:
        print("\nExiting Docker container...")
    except Exception as e:
        print(f"Error running Docker container: {e}")


def run_docker_with_app(docker_cmd):
    """Run Docker container with CLI chat local application."""
    docker_cmd.extend([
        "bash",
        "-c",
        "cd /ThirdParty/InServiceOfX/PythonApplications/CLIChatLocal/Executables && python main_CLIChatLocal.py --dev"
    ])
    
    try:
        subprocess.run(docker_cmd)
    except KeyboardInterrupt:
        print("\nExiting Docker container...")
    except Exception as e:
        print(f"Error running Docker container: {e}")


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

    # Read database network
    database_network = read_database_docker_compose(build_dir)
    print(f"Using database network: {database_network}")

    # Build Docker command
    docker_cmd = build_docker_command(
        image_name,
        volumes,
        args.gpu,
        args.port,
        database_network
    )
    
    print("\nExecuting Docker command:")
    print(" ".join(docker_cmd))
    print("\n")
    
    # Execute the command with either shell or app
    if args.clichatlocal:
        run_docker_with_app(docker_cmd)
    else:
        run_docker_with_shell(docker_cmd)


if __name__ == "__main__":
    main()
