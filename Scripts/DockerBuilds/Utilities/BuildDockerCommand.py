from CommonUtilities import run_command

class BuildDockerCommand:

    DEFAULT_BUILD_ARGUMENTS = ["base_image", "docker_image_name"]

    def __init__(self, build_arguments_keys = None):
        """
        Args:
            build_arguments_keys (list): List of build arguments keys.
        """
        if build_arguments_keys is None:
            build_arguments_keys = []
        self._build_arguments_keys = build_arguments_keys

    def run_build_docker_command(
        self,
        dockerfile_path,
        build_configuration,
        use_cache,
        build_context,
        use_host_network=False):
        """
        Builds the Docker image using the provided Dockerfile and build arguments.
        
        Args:
            dockerfile_path (Path): Path to the Dockerfile.
            build_configuration: Typically result from BuildDockerConfiguration.
            use_cache (bool): Whether to use Docker cache during build.
            build_context (Path): The directory to use as the build context.
            use_host_network (bool): Whether to use --network host during build.

        Raises:
            subprocess.CalledProcessError: If the Docker build command fails.
            ValueError: If the BASE_IMAGE is empty in the configuration.
        """
        docker_build_cmd = ["DOCKER_BUILDKIT=1", "docker", "build"]

        if not use_cache:
            docker_build_cmd.append("--no-cache")
        
        # Add network host if requested
        if use_host_network:
            docker_build_cmd.append("--network host")

        # Add build arguments
        for key in self._build_arguments_keys:
            docker_build_cmd.extend([
                "--build-arg",
                f"{key}={build_configuration[key]}"])

        # Check and add BASE_IMAGE argument
        base_image = build_configuration.base_image
        if not base_image:
            raise ValueError("BASE_IMAGE is empty in the configuration file")
        docker_image_name = build_configuration.docker_image_name

        docker_build_cmd.extend([
            "--build-arg",
            f"BASE_IMAGE={base_image}"
        ])

        # Specify Dockerfile
        docker_build_cmd.extend(["-f", str(dockerfile_path)])

        # Tag the image
        docker_build_cmd.extend(["-t", docker_image_name])

        docker_build_cmd.append(".")

        # Convert command list to string
        command_str = ' '.join(docker_build_cmd)

        run_command(command_str, cwd=build_context)