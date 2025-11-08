from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import yaml

@dataclass
class VolumeMount:
    """Configuration for a volume mount."""
    host_path: str
    container_path: str

@dataclass
class PortMapping:
    """Configuration for a port mapping."""
    host_port: int
    container_port: int

@dataclass
class RunDockerConfigurationData:
    volumes: Optional[List[VolumeMount]] = None
    ports: Optional[List[PortMapping]] = None
    
    def __post_init__(self):
        """Convert dict data to proper dataclass instances."""
        if self.volumes is None:
            self.volumes = []
        elif isinstance(self.volumes, list):
            # Convert dict list to VolumeMount objects
            converted_volumes = []
            for vol in self.volumes:
                if isinstance(vol, dict):
                    converted_volumes.append(VolumeMount(**vol))
                else:
                    converted_volumes.append(vol)
            self.volumes = converted_volumes
        
        if self.ports is None:
            self.ports = []
        elif isinstance(self.ports, list):
            # Convert dict list to PortMapping objects
            converted_ports = []
            for port in self.ports:
                if isinstance(port, dict):
                    converted_ports.append(PortMapping(**port))
                else:
                    converted_ports.append(port)
            self.ports = converted_ports

class RunDockerConfiguration:
    DEFAULT_FILE_NAME = "run_docker_configuration.yml"
    DEFAULT_FILE_DIR = Path.cwd().parent
    
    @staticmethod
    def load_data(file_path: Optional[Union[Path, str]] = None) \
        -> RunDockerConfigurationData:
        if file_path is None:
            file_path = \
                RunDockerConfiguration.DEFAULT_FILE_DIR / \
                    RunDockerConfiguration.DEFAULT_FILE_NAME
        file_path = Path(file_path)
        
        # Return default if file doesn't exist (optional config file)
        if not file_path.exists():
            return RunDockerConfigurationData()
        
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        
        # Handle empty file
        if data is None:
            data = {}
        
        return RunDockerConfigurationData(**data)
