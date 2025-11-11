from pathlib import Path
from typing import List,Optional, Union
from warnings import warn
import subprocess
import yaml

class DockerCompose:
    def __init__(
        self,
        docker_compose_file_path: Optional[Union[str, Path]] = None):
        if docker_compose_file_path == None:
            docker_compose_file_path = \
                Path.cwd().resolve().parent / "docker-compose.yml"

        self._docker_compose_file_path = docker_compose_file_path
        self._networks = None

    def parse_networks(self) -> Optional[List[str]]:
        """
        Parse network names from docker-compose.yml.
        
        Returns:
            List of network name strings, or None if no networks found.
            Network names are extracted from the 'name' field if present,
            otherwise the key is used.
        """
        if self._docker_compose_file_path.exists():
            with open(self._docker_compose_file_path, 'r') as f:
                docker_compose_configuration = yaml.safe_load(f)
                if 'networks' in docker_compose_configuration:
                    network_names = []
                    for key, network_config \
                        in docker_compose_configuration['networks'].items():
                        # If network_config is a dict with a 'name' field, use
                        # that. Otherwise, use the key itself
                        if isinstance(network_config, dict) and \
                            'name' in network_config:
                            network_names.append(network_config['name'])
                        else:
                            # Use the key as the network name
                            network_names.append(key)
                    return network_names if network_names else None
                else:
                    return None
        else:
            warn(
                "Docker compose.yml doesn't exist: ",
                str(self._docker_compose_file_path))
            return None

    def run_docker_compose(self):
        if self._docker_compose_file_path.exists():
            try:
                subprocess.run(
                    [
                        'docker',
                        'compose',
                        '-f',
                        str(self._docker_compose_file_path),
                        'up',
                        '-d'],
                    check=True
                )
            except Exception as err:
                warn(f'Warning: Failed to start docker-compose: {err}')
        else:
            warn(
                "Docker compose.yml doesn't exist: ",
                str(self._docker_compose_file_path)
            )
