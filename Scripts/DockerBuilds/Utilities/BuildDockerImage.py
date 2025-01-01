from CommonUtilities import run_command

class BuildDockerImage:

    def __init__(self, build_arguments_keys):
        """
        Args:
            build_arguments_keys (list): List of build arguments keys.

            Note that build_arguments_keys does NOT have to include BASE_IMAGE
            (for what base image to "base" this Docker image upon) and
            DOCKER_IMAGE_NAME (the name of the Docker image) because the are
            assumed to exist and are hardcoded to be accessed in the
            build_docker_image(..) function.
        """
        self.build_arguments_keys = build_arguments_keys

    def build_docker_image(
        self,
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
            ValueError: If the BASE_IMAGE is empty in the configuration.
        """
        docker_build_cmd = ["DOCKER_BUILDKIT=1", "docker", "build"]

        if not use_cache:
            docker_build_cmd.append("--no-cache")

        # Add build arguments
        for key in self.build_arguments_keys:
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

        # Specify Dockerfile
        docker_build_cmd.extend(["-f", str(dockerfile_path)])

        # Tag the image
        docker_build_cmd.extend(["-t", docker_image_name])

        docker_build_cmd.append(".")

        # Convert command list to string
        command_str = ' '.join(docker_build_cmd)

        run_command(command_str, cwd=build_context)

class BuildDockerImageWithNVIDIAGPU(BuildDockerImage):
    def __init__(self):
        super().__init__(["ARCH", "PTX", "COMPUTE_CAPABILITY"])
