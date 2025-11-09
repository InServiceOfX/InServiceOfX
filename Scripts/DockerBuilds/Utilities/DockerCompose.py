from pathlib import Path
from typing import Optional, Union
from warnings import warn
import subprocess
import yaml

class DockerCompose:
    def __init__(
        self,
        docker_compose_file_path = Optional[Union[str, Path]] = None):
        if docker_compose_file_path == None:
            docker_compose_file_path = \
                Path.cwd().resolve().parent / "docker-compose.yml"

        self._docker_compose_file_path = docker_compose_file_path
        self._networks = None

    def parse_networks(self):
        if self._docker_compose_file_path.exists():
            with open(self._docker_compose_file_path, 'r') as f:
                docker_compose_configuration = yaml.safe_load(f)
                if 'networks' in docker_compose_configuration:
                    self._networks = {}
                    for key in docker_compose_configuration['networks'].keys():
                        self._networks[key] = \
                            docker_compose_configuration['networks'][key]
                    return self._networks
                else:
                    return None
        else:
            warn(
                "Docker compose.yml doesn't exist: ",
                str(self._docker_compose_file_path))

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