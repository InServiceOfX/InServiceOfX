#!/usr/bin/env python3

import sys
from pathlib import Path

docker_builds_directory = Path(__file__).resolve().parents[4]

if docker_builds_directory.exists():
    if str(docker_builds_directory) not in sys.path:
        sys.path.append(str(docker_builds_directory))
else:
    raise FileNotFoundError(
        f"Docker builds directory {docker_builds_directory} does not exist")

from Utilities import (BuildDockerBaseClass, BuildDockerCommand,)
from Utilities.BuildDockerConfiguration import BuildDockerConfiguration
from typing import List, Tuple

class BuildDocker(BuildDockerBaseClass):
    def __init__(self):

        self._build_directory = Path(__file__).resolve().parent
        build_configuration = BuildDockerConfiguration.load_data(
            self._build_directory / "build_configuration.yml")
        super().__init__(
            "Build Docker image for full music development",
            build_configuration,
            self._build_directory,
            BuildDockerCommand(),
            3)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self._parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.minimal_base",
                self._parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.cpp",
                self._parent_dir / "CommonFiles" / \
                    "Dockerfile.cpp_development"),
            (
                "Dockerfile.transformers",
                self._build_directory / "Dockerfile.transformers"),
            (
                "Dockerfile.third_parties",
                self._build_directory / "Dockerfile.third_parties"),
            (
                "Dockerfile.apis",
                self._build_directory / "Dockerfile.apis"),
            (
                "Dockerfile.music",
                self._build_directory / "Dockerfile.music"),
        ]

def main():
    build_docker = BuildDocker()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()