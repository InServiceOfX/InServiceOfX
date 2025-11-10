from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DockerRunConfiguration:
    """Configuration for running Docker containers."""
    docker_image_name: str
    volumes: list = None  # Changed from mount_paths to volumes
    ports: list = None  # New field
    gpu_id: Optional[int] = None
    interactive: bool = True
    entrypoint: Optional[str] = None
    use_host_network: bool = False
    networks: Optional[List[str]] = None

    def __post_init__(self):
        if self.volumes is None:
            self.volumes = []
        if self.ports is None:
            self.ports = []
        if self.networks is None:
            self.networks = []
        elif not isinstance(self.networks, list):
            self.networks = [self.networks]


class DockerRunCommandBuilder:
    """Builds docker run commands from configuration."""
    
    def __init__(self, config: DockerRunConfiguration):
        self.config = config
    
    def build(self) -> list:
        """Build complete docker run command as list."""
        cmd = ["docker", "run"]
        
        # GPU configuration
        if self.config.gpu_id is not None:
            cmd.extend(["--gpus", f"device={self.config.gpu_id}"])
        else:
            cmd.extend(["--gpus", "all"])
        
        # Interactive mode with TTY
        if self.config.interactive:
            cmd.extend(["-it"])
        else:
            cmd.append("-d")  # Detached mode
        
        # Network host option (if enabled, skip port mappings)
        if self.config.use_host_network:
            cmd.append("--network host")
        else:
            # Port mappings (only apply if not using host network)
            for port in self.config.ports:
                cmd.append(f"-p {port['host_port']}:{port['container_port']}")

        if self.config.networks:
            for network in self.config.networks:
                cmd.append(f"--network {network}")

        # Mount paths from configuration (always needed)
        for mount in self.config.volumes:
            cmd.append(f"-v {mount['host_path']}:{mount['container_path']}")
        
        # Environment variables for NVIDIA runtime
        # Don't set CUDA_VISIBLE_DEVICES - let Docker handle GPU filtering
        cmd.extend([
            "-e", "NVIDIA_DISABLE_REQUIRE=1",
            "-e", "CUDA_VISIBLE_DEVICES=0",  # When --gpus device=X is used, inside container it's device 0
        ])
        
        # Runtime flags
        cmd.extend(["--rm", "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"])
        
        # Entrypoint override if specified
        if self.config.entrypoint:
            cmd.extend(["--entrypoint", self.config.entrypoint])
        
        # Image name
        cmd.append(self.config.docker_image_name)
        
        return cmd
