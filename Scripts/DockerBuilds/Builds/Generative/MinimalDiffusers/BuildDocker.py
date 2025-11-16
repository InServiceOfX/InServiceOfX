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

class BuildDockerNew(BuildDockerBaseClass):
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
                "Dockerfile.essential_pip_installs",
                self._build_directory / "Dockerfile.essential_pip_installs"),
            (
                "Dockerfile.diffusers",
                self._build_directory / "Dockerfile.diffusers"),
        ]

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for diffusers with minimal dependencies",
            Path(__file__).resolve(),
            3,
            configuration_reader_class=ReadBuildConfigurationWithOpenCV,
            docker_builder_class=BuildDockerImageWithOpenCV)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.minimal_base",
                self.parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.more_pip_installs",
                self.script_dir / "Dockerfile.more_pip_installs"),
            (
                "Dockerfile.opencv_with_cuda",
                self.parent_dir / "CommonFiles" / "Dockerfile.opencv_with_cuda"),
            (
                "Dockerfile.diffusers",
                self.script_dir / "Dockerfile.diffusers"),
            (
                 "Dockerfile.third_parties",
                 self.script_dir / "Dockerfile.third_parties")
        ]

def main():
    build_docker = BuildDockerNew()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()