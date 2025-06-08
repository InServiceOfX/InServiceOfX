class CreateDockerRunCommand:
    def __init__(
            self,
            project_directory,
            build_configuration,
            configuration,
            gpu_id: int = None):
        """
        Initialize docker run command creator
        
        Args:
            project_directory (Path): Project root directory
            build_configuration (dict): Build configuration settings
            configuration (dict): Run configuration settings
            gpu_id (int, optional): Specific GPU ID to use. If None, uses all GPUs.
        """
        self.docker_image_name = build_configuration["DOCKER_IMAGE_NAME"]
        self.configuration = configuration
        self.gpu_id = gpu_id

        self.docker_run_command = self.create_docker_run_command(
            project_directory,
            self.configuration,
            self.docker_image_name)

    def create_docker_run_command(
        self,
        project_directory,
        configuration,
        docker_image_name
        ):
        """
        Create docker run command with optional GPU selection
        
        Args:
            project_directory (Path): Project root directory
            mount_paths (list): List of volume mounts
            docker_image_name (str): Name of the docker image
            
        Returns:
            str: Complete docker run command
        """
        # Base command with interactive TTY
        docker_run_command = "docker run "
        
        # Add project directory mount
        docker_run_command += f"-v {project_directory}:/InServiceOfX "
        
        # Configure GPU access
        if self.gpu_id is not None:
            # Use specific GPU
            docker_run_command += f"--gpus device={self.gpu_id} "
        else:
            # Use all available GPUs
            docker_run_command += "--gpus all "

        # Add interactive TTY flags:
        # -it - i stands for interactive, so this flag makes sure that standard
        # input ('STDIN') remains open even if you're not attached to container.
        # -t stands for pseudo-TTY, allocates a pseudo terminal inside
        # container, used to make environment inside container feel like a
        # regular shell session.
        docker_run_command += "-it "

        # Add mount paths from configuration file
        if "mount_paths" in configuration:
            for mount_path in configuration["mount_paths"]:
                docker_run_command += f"-v {mount_path} "

        # Enable CUDA Forward Compatibility
        docker_run_command += "-e NVIDIA_DISABLE_REQUIRE=1 "

        if "ports" in configuration:
            for port in configuration["ports"]:
                docker_run_command += f"-p {port}:{port} "

        # Add ports for gradio and jupyter
        docker_run_command += "-p 8888:8888 -p 7860:7860 --rm --ipc=host "

        # Add network if specified
        if "network" in configuration:
            docker_run_command += f"--network {configuration['network']} "

        # Add image name and default command
        docker_run_command += f"{docker_image_name}"

        print(f"Docker run command: {docker_run_command}")

        return docker_run_command