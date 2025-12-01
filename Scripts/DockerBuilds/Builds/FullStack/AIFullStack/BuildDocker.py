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

from typing import List, Tuple

class BuildDocker(BuildDockerBaseClass):
    def __init__(self):

        self._build_directory = Path(__file__).resolve().parent
        build_configuration = BuildDockerConfiguration.load_data(
            self._build_directory / "build_configuration.yml")
        super().__init__(
            "Build Docker image for generative AI full-stack applications; no CUDA",
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
                "Dockerfile.base",
                self._parent_dir / "CommonFiles" / "Dockerfile.base"),
            (
                "Dockerfile.more_apt_installs",
                self._build_directory / "Dockerfile.more_apt_installs"),
            (
                "Dockerfile.cpp",
                self._parent_dir / "CommonFiles" / \
                    "Dockerfile.cpp_development"),
            (
                "Dockerfile.python",
                self._parent_dir / "CommonFiles" / "Dockerfile.python"),
            (
                "Dockerfile.rust",
                self._parent_dir / "CommonFiles" / "Dockerfile.rust"),
            (
                "Dockerfile.nvm_latest",
                self._build_directory / "Dockerfile.nvm_latest"),
            (
                "Dockerfile.more_pip_installs",
                self._build_directory / "Dockerfile.more_pip_installs"),
        ]

def main():
    build_docker = BuildDocker()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()