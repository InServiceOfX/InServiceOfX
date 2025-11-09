"""
Run Docker container with general support for any Docker build configuration.

Usage: 
    python RunDocker.py [--build-dir DIR] [--gpu GPU_ID] [--interactive] [--entrypoint ENTRYPOINT]

This script loads the build configuration from a specified build directory and runs
the built Docker image with appropriate GPU settings and volume mounts.
"""

from pathlib import Path
import sys

docker_builds_dir = Path(__file__).resolve().parents[1]
if (str(docker_builds_dir) not in sys.path):
    sys.path.append(str(docker_builds_dir))

from Utilities.BuildDockerConfiguration import BuildDockerConfiguration
from Utilities.RunDockerConfiguration import (
    PortMapping,
    RunDockerConfigurationData,
    RunDockerConfiguration,
    VolumeMount)    
from Utilities.DockerRunCommandBuilder import (
    DockerRunConfiguration,
    DockerRunCommandBuilder)
from Utilities import DockerCompose

import argparse
import os
import subprocess

from CommonUtilities import run_command

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Docker container with build configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all GPUs in interactive mode (default, from current directory)
  python RunDocker.py
  
  # Specify build directory
  python RunDocker.py --build-dir ../OmegaTensorRT-LLM
  
  # Run with specific GPU
  python RunDocker.py --gpu 0
  
  # Run in non-interactive (detached) mode
  python RunDocker.py --no-interactive
  
  # Override entrypoint to exec into a shell
  python RunDocker.py --entrypoint /bin/bash
  
  # Use host networking (bypasses Docker port mapping, default True)
  python RunDocker.py --network-host
        """
    )
    
    parser.add_argument(
        '--build-dir',
        type=str,
        default='.',
        help='Directory containing build_docker_configuration.yml (default: current directory)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        help='Specific GPU ID to use (0, 1, 2, etc.). If not specified, uses all GPUs.'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Run in detached (background) mode instead of interactive'
    )
    
    parser.add_argument(
        '--entrypoint',
        type=str,
        help='Override the entrypoint (e.g., /bin/bash to get a shell)'
    )
    
    parser.add_argument(
        '--network-host',
        action='store_true',
        help='Use host networking (--network host). Bypasses Docker port mapping. Useful when experiencing networking issues.'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Determine build directory
    if args.build_dir == '.':
        build_dir = Path.cwd()
    else:
        build_dir = Path(args.build_dir).resolve()
    
    # Find build configuration file
    config_file = build_dir / "build_configuration.yml"
    
    if not config_file.exists():
        print(f"Error: Build configuration file not found: {config_file}")
        print(f"  Searched in: {build_dir}")
        sys.exit(1)
    
    # Load configuration
    try:
        build_config = BuildDockerConfiguration.load_data(config_file)
        print(f"Loaded configuration from: {config_file}")
        print(f"Docker image: {build_config.docker_image_name}")
        print(f"Base image: {build_config.base_image}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check if image exists
    result = subprocess.run(
        ["docker", "images", "-q", build_config.docker_image_name],
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        print(f"Warning: Docker image '{build_config.docker_image_name}' not found.")
        print("You may need to run BuildDocker.py first to build the image.")
        
        if not args.no_interactive:
            print("Do you want to continue anyway? (y/n): ", end="")
            response = input().strip().lower()
            if response != 'y':
                sys.exit(0)
    
    # Load run configuration (optional)
    run_config_file = build_dir / "run_docker_configuration.yml"
    try:
        run_config_data = RunDockerConfiguration.load_data(run_config_file)
    except Exception as e:
        print(f"Warning: Could not load run configuration: {e}")
        run_config_data = RunDockerConfigurationData()

    run_config = DockerRunConfiguration(
        docker_image_name=build_config.docker_image_name,
        volumes=[{"host_path": v.host_path, "container_path": v.container_path} for v in run_config_data.volumes],
        ports=[{"host_port": p.host_port, "container_port": p.container_port} for p in run_config_data.ports],
        gpu_id=args.gpu,
        interactive=not args.no_interactive,
        entrypoint=args.entrypoint,
        use_host_network=args.network_host,
    )
    
    # Build docker run command
    command_builder = DockerRunCommandBuilder(run_config)
    docker_cmd_list = command_builder.build()
    
    # Convert to string for display and execution
    docker_command_str = " ".join(docker_cmd_list)
    
    print(f"\nDocker command to execute:")
    print(f"  {docker_command_str}\n")
    print("=" * 80)
    
    # Execute the command using subprocess for better visibility
    print("Starting container...\n")
    
    # Use shell=True since we have a proper string command
    run_command(
        docker_command_str, 
        cwd=build_dir)

if __name__ == "__main__":
    main()