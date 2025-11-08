from CommonUtilities import run_command

class BuildDockerCommand:

    DEFAULT_BUILD_ARGUMENTS = ["base_image", "docker_image_name"]

    @staticmethod
    def run_build_docker_command(
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

        # Extract build_args from build_configuration
        # build_args is a Dict[str, str] from the YAML file
        build_args = getattr(build_configuration, 'build_args', {})
        if build_args is None:
            build_args = {}

        # Add ALL build arguments from build_args dict to Docker build command
        # This makes the configuration fully dynamic - any key-value pairs
        # in the YAML's build_args section will be passed to Docker
        for key, value in build_args.items():
            docker_build_cmd.extend([
                "--build-arg",
                f"{key}={value}"])

        # Check and add BASE_IMAGE argument
        # Access base_image as an attribute (from BuildDockerConfigurationData)
        base_image = getattr(build_configuration, 'base_image', None)
        if not base_image:
            raise ValueError("BASE_IMAGE is empty in the configuration file")
        
        docker_image_name = getattr(
            build_configuration,
            'docker_image_name',
            None)
        if not docker_image_name:
            raise ValueError(
                "DOCKER_IMAGE_NAME is empty in the configuration file")

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