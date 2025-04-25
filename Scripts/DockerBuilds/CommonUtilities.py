from pathlib import Path
import sys
import subprocess


def run_command(command, cwd=None):
    """
    Runs a shell command and captures its output.

    Args:
        command (str): The command to execute.
        cwd (Path, optional): The working directory to execute the command in.

    Returns:
        subprocess.CompletedProcess: The result of the executed command.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            status.
    """
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            # Commented out to try to stream output directly to console.
            #check=True,
            text=True,
            cwd=cwd)
        if result.returncode != 0:
            print(
                f"Error: Command '{command}' failed with exit code {result.returncode}",
                file=sys.stderr)
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as err:
        if err.returncode == 126:
            print(
                f"Error: Permission denied. Cannot run the command: {command}",
                file=sys.stderr)
        else:
            print(
                f"Error: Command '{command}' failed with exit code {err.returncode}",
                file=sys.stderr)
        sys.exit(err.returncode)

class DefaultValues:
    BUILD_FILE_NAME = "build_docker_configuration.txt"
    RUN_CONFIGURATION_FILE_NAME = "run_docker_configuration.txt"


def get_project_directory():
    # Resolve to absolute path.
    current_filepath = Path(__file__).resolve()

    # This variable's value depends on the location of this file relative to
    # other subdirectories. It depends on the location of this file, even if
    # this function is imported and used in another file in another
    # subdirectory.
    parent_subdirectories = 2

    project_directory = current_filepath.parents[parent_subdirectories]

    if not project_directory.is_dir():
        print(f"{project_directory} is not a directory")
        exit(1)

    if not project_directory.exists():
        print(f"{project_directory} is not an existing directory")
        exit(1)

    return project_directory

def get_docker_builds_directory():
    return get_project_directory() / "Scripts" / "DockerBuilds"

def concatenate_dockerfiles(output_dockerfile, *dockerfile_paths):
    """
    Concatenates multiple Dockerfile components into a single Dockerfile.

    Args:
        output_dockerfile (Path): Path where the concatenated Dockerfile will be
        saved.
        *dockerfile_paths (Path): Arbitrary number of paths to Dockerfiles to
        concatenate.

    Raises:
        FileNotFoundError: If any of the input Dockerfile components are
        missing.
    """
    with output_dockerfile.open('w') as outfile:
        for file_path in dockerfile_paths:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Dockerfile component '{file_path}' does not exist.")
            with file_path.open('r') as infile:
                outfile.write(infile.read())
# Ensure separation between files
                outfile.write('\n')

    print(f"Successfully concatenated Dockerfiles into '{output_dockerfile}'.")


def build_docker_image_for_arm64(
    dockerfile_path,
    build_configuration,
    use_cache,
    build_context):
    """
    Builds the Docker image using the provided Dockerfile and build arguments.

    Args:
        dockerfile_path (Path): Path to the Dockerfile.
        build_configuration: Typically result from read_build_configuration.
        use_cache (bool): Whether to use Docker cache during build.
        build_context (Path): The directory to use as the build context.

    Raises:
        subprocess.CalledProcessError: If the Docker build command fails.
        ValueError: If the BASE_IMAGE or ARM64_BASE_IMAGE is empty in the configuration.
    """
    # See https://docs.docker.com/build/buildkit/
    docker_build_cmd = ["DOCKER_BUILDKIT=1", "docker", "buildx", "build"]

    if not use_cache:
        docker_build_cmd.append("--no-cache")

    build_argument_keys = ["ARCH", "PTX", "COMPUTE_CAPABILITY"]

    # Add build arguments
    for key in build_argument_keys:
        docker_build_cmd.extend([
            "--build-arg",
            f"{key}={build_configuration[key]}"])

    # Check and add BASE_IMAGE argument
    base_image = build_configuration.get('BASE_IMAGE', '')
    if not base_image:
        raise ValueError("BASE_IMAGE is empty in the configuration file")
    docker_image_name = build_configuration['DOCKER_IMAGE_NAME']

    docker_build_cmd.extend([
        "--build-arg",
        f"BASE_IMAGE={base_image}"
    ])

    # Specify platform if building for ARM 64
    docker_build_cmd.extend(["--platform", "linux/arm64"])
    # Add --output option for ARM64 builds
    output_file = f"{docker_image_name}.tar"
    docker_build_cmd.extend(["--output", f"type=tar,dest={output_file}"])

    # Specify Dockerfile
    docker_build_cmd.extend(["-f", str(dockerfile_path)])

    # Tag the image
    docker_build_cmd.extend(["-t", docker_image_name])

    # Specify build context
    docker_build_cmd.append(str(build_context))

    # Convert command list to string
    command_str = ' '.join(docker_build_cmd)

    run_command(command_str, cwd=build_context)


def parse_run_configuration_file(configuration_file_path):
    configuration = {}
    mount_paths = []
    ports = []

    if configuration_file_path.exists():
        with open(configuration_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Skip comments and empty lines
                if stripped_line.startswith('#') or not stripped_line:
                    continue

                if '=' in stripped_line:
                    key, value = stripped_line.split('=', 1)
                    key = key.strip()

                    if key.startswith("EXPOSE_PORT_") or \
                        key.startswith("EXPOSE_PORT"):
                        try:
                            port = int(value.strip())
                            ports.append(port)
                        except ValueError:
                            print(f"Invalid port number: {value}")

                    if key.startswith("MOUNT_PATH_") or \
                        key.startswith("MOUNT_PATH"):

                        print(value)
                        print(Path(value.split(':', 1)[0]).resolve())

                        if ':' in value and \
                            Path(value.split(':', 1)[0]).resolve().is_dir():
                            mount_paths.append(value)

                        else:
                            print(
                                f"Invalid or nonexistent path in file: {line.split(':', 1)[0]}")
    else:
        print(f"Configuration file '{configuration_file_path}' does not exist.")

    configuration['mount_paths'] = mount_paths
    configuration['ports'] = ports
    return configuration

class CreateDockerRunCommand:
    def __init__(
            self,
            project_directory,
            build_configuration,
            configuration,
            is_arm64):
        self.is_arm64 = is_arm64
        self.docker_image_name = \
            build_configuration["ARM64_DOCKER_IMAGE_NAME"] if is_arm64 \
                else build_configuration["DOCKER_IMAGE_NAME"]
        self.mount_paths = configuration["mount_paths"]

        self.docker_run_command = self.create_docker_run_command(
            project_directory,
            self.mount_paths,
            self.docker_image_name,
            self.is_arm64)

    def create_docker_run_command(
        self,
        project_directory,
        mount_paths,
        DOCKER_IMAGE_NAME,
        is_arm64
        ):
        # Run command
        # -it - i stands for interactive, so this flag makes sure that standard
        # input ('STDIN') remains open even if you're not attached to container.
        # -t stands for pseudo-TTY, allocates a pseudo terminal inside
        # container, used to make environment inside container feel like a
        # regular shell session.
        gpu_option = "--runtime=nvidia" if is_arm64 else "--gpus all"
        docker_run_command = \
            f"docker run -v {project_directory}:/InServiceOfX {gpu_option} -it "

        # Add mount paths from configuration file
        for mount_path in mount_paths:
            docker_run_command += f"-v {mount_path} "

        # -e flag sets environment and enables CUDA Forward Compatibility instead of
        # default CUDA Minor Version Compatibility.
        docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

        # Add the port 7860 for gradio applications.
        docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host "

        docker_run_command += DOCKER_IMAGE_NAME

        if is_arm64:
            docker_run_command += " /bin/bash"

        print(docker_run_command)

        return docker_run_command
