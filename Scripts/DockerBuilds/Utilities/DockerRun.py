"""
Run Docker container with general support for any Docker build configuration.

Usage: 
    python RunDocker.py [--build-dir DIR] [--gpu GPU_ID] [--interactive] [--entrypoint ENTRYPOINT]

Example Usage:
cd into "Build" directory, e.g. cd DockerBuilds/Builds/LLM/TensorRTLLMBased
python ../../../Utilities/DockerRun.py --gpu 1 --network-host --build-dir .

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

def check_and_setup_x11():
    """Check if xhost is configured for Docker, and set it up if needed."""
    try:
        # Check if xhost already allows Docker
        result = subprocess.run(
            ["xhost"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # TODO: Check if finding "LOCAL:" is sufficient.
        if "LOCAL:docker" in result.stdout:
            print("✓ X11 access already configured for Docker")
            return True
        else:
            print("⚠ X11 access not configured for Docker")
            print("  Setting up X11 access...")
            result = subprocess.run(
                ["xhost", "+local:docker"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ X11 access enabled for Docker")
                print("  (This resets on reboot/logout)")
                return True
            else:
                print(f"⚠ Warning: Failed to set up X11 access: {result.stderr}")
                print("  You may need to run: xhost +local:docker")
                return False
    except FileNotFoundError:
        print("⚠ Warning: 'xhost' command not found. X11 may not work.")
        print("  Make sure you're running from a graphical session.")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Warning: xhost command timed out")
        return False
    except Exception as e:
        print(f"⚠ Warning: Could not check X11 setup: {e}")
        return False

def check_audio_setup():
    """Check if audio is available on the host and provide helpful
    information."""
    audio_available = False
    pulse_available = False
    alsa_available = False

    # Check PulseAudio (Ubuntu Desktop default)
    pulse_socket = f"/run/user/{os.getuid()}/pulse/native"
    if os.path.exists(pulse_socket):
        pulse_available = True
        audio_available = True
        print("✓ PulseAudio detected (recommended for Ubuntu Desktop)")

    # Check ALSA as fallback
    if Path("/dev/snd").exists():
        alsa_available = True
        if not audio_available:
            audio_available = True
            print("⚠ PulseAudio not found, will use ALSA (may have limitations)")

    if not audio_available:
        print("⚠ Warning: No audio devices found")
        print("  PulseAudio socket not found and /dev/snd not available")
        print("  Audio may not work in the container")
        return False

    # Check if PulseAudio is running (if socket exists)
    if pulse_available:
        try:
            result = subprocess.run(
                ["pulseaudio", "--check"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print("✓ PulseAudio is running")
            else:
                print(
                    "⚠ PulseAudio socket exists but daemon may not be running")
                print("  Try: pulseaudio --start")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # pulseaudio command not found or timed out, but socket exists
            # This is okay - the socket might work anyway
            pass
    
    return True

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

  # Run with no GPU
  python RunDocker.py --no-gpu

  # Enable GUI support (X11 forwarding) for applications like Mixxx
  python RunDocker.py --gui

  # Enable audio support (PulseAudio + ALSA)
  python RunDocker.py --audio

  # Enable both GUI and audio (common for applications like Mixxx)
  python RunDocker.py --gui --audio
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

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Run with no GPU'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable GUI support (X11 forwarding)'
    )

    parser.add_argument(
        '--audio',
        action='store_true',
        help=(
            'Enable audio support (PulseAudio + ALSA). Requires PulseAudio on '
            'host for best results.')
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup X11 if GUI is requested
    if args.gui:
        if not check_and_setup_x11():
            print("\n⚠ Warning: X11 setup may have failed.")
            print("  The container will still run, but GUI applications may not work.")
            print("  You can manually run: xhost +local:docker\n")

    # Check audio setup if audio is requested
    if args.audio:
        print("\nChecking audio setup...")
        if not check_audio_setup():
            print("\n⚠ Warning: Audio setup check failed.")
            print("  The container will still run, but audio may not work.")
            print("  For PulseAudio, ensure it's running: pulseaudio --start\n")

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

    # Find docker-compose.yml file
    docker_compose_file = build_dir / "docker-compose.yml"
    if docker_compose_file.exists():
        docker_compose = DockerCompose(docker_compose_file)
        networks = docker_compose.parse_networks()
        docker_compose.run_docker_compose()
    else:
        networks = None

    # Load run configuration (optional)
    run_config_file = build_dir / "run_configuration.yml"

    if not run_config_file.exists():
        print(f"Warning: Run configuration file not found: {run_config_file}")
        print(f"  Searched in: {build_dir}")
    else:
        print(f"Loading run configuration from: {run_config_file}")

    try:
        run_config_data = RunDockerConfiguration.load_data(run_config_file)
    except Exception as e:
        print(f"Warning: Could not load run configuration: {e}")
        run_config_data = RunDockerConfigurationData()

    run_config = DockerRunConfiguration(
        docker_image_name=build_config.docker_image_name,
        volumes=[
            {"host_path": v.host_path, "container_path": v.container_path} \
                for v in run_config_data.volumes],
        ports=[
            {"host_port": p.host_port, "container_port": p.container_port} \
                for p in run_config_data.ports],
        gpu_id=args.gpu,
        interactive=not args.no_interactive,
        entrypoint=args.entrypoint,
        use_host_network=args.network_host,
        networks=networks,
        enable_gui=args.gui,
        enable_audio=args.audio
    )
    
    # Build docker run command
    command_builder = DockerRunCommandBuilder(run_config)

    if args.no_gpu:
        docker_cmd_list = command_builder.build_with_no_gpu()
    else:
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