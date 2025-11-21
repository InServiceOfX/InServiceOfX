#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

docker_builds_directory = Path(__file__).resolve().parents[3]

if docker_builds_directory.exists():
    if str(docker_builds_directory) not in sys.path:
        sys.path.append(str(docker_builds_directory))
else:
    raise FileNotFoundError(
        f"Docker builds directory {docker_builds_directory} does not exist")

from Utilities import (BuildDockerBaseClass, BuildDockerCommand,)

from Utilities.BuildDockerConfiguration import BuildDockerConfiguration

from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationWithOpenCV,
    BuildDockerImageWithOpenCV)

from typing import List, Tuple

class BuildDocker(BuildDockerBaseClass):
    def __init__(self):

        self._build_directory = Path(__file__).resolve().parent
        build_configuration = BuildDockerConfiguration.load_data(
            self._build_directory / "build_configuration.yml")
        super().__init__(
            "Build Docker image for diffusers with minimal dependencies or at least no OpenCV",
            build_configuration,
            self._build_directory,
            BuildDockerCommand(),
            2)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self._parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.minimal_base",
                self._parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.pytorch_reinstall",
                self._build_directory / "Dockerfile.pytorch_reinstall"),
            (
                "Dockerfile.nunchaku",
                self._build_directory / "Dockerfile.nunchaku"),
        ]

def main():
    build_docker = BuildDocker()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()