#!/usr/bin/env python3

import sys
from pathlib import Path

docker_builds_directory = Path(__file__).resolve().parents[3]

if docker_builds_directory.exists():
    if str(docker_builds_directory) not in sys.path:
        sys.path.append(str(docker_builds_directory))
else:
    raise FileNotFoundError(
        f"Docker builds directory {docker_builds_directory} does not exist")

from Utilities import (
    BuildDockerBase,
    BuildDockerBaseClass,
    BuildDockerCommand,
    ReadBuildConfigurationWithNunchaku,
    BuildDockerImageWithNunchaku)
from Utilities.BuildDockerConfiguration import BuildDockerConfiguration

from typing import List, Tuple

class BuildDockerNew(BuildDockerBaseClass):
    def __init__(self):

        self._build_directory = Path(__file__).resolve().parent
        build_configuration = BuildDockerConfiguration.load_data(
            self._build_directory / "build_configuration.yml")
        super().__init__(
            "Build Docker image for diffusers",
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
        ]

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for diffusers with minimal dependencies",
            Path(__file__).resolve(),
            3,
            configuration_reader_class=ReadBuildConfigurationWithNunchaku,
            docker_builder_class=BuildDockerImageWithNunchaku)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.minimal_base",
                self.parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.essential_pip_installs",
                self.script_dir / "Dockerfile.essential_pip_installs"),
            # (
            #     "Dockerfile.copy_opencv_script",
            #     self.script_dir / "Dockerfile.copy_opencv_script"),
            # (
            #     "Dockerfile.opencv_with_cuda",
            #     self.parent_dir / "CommonFiles" / "Dockerfile.opencv_with_cuda"),
            # (
            #     "Dockerfile.diffusers",
            #     self.script_dir / "Dockerfile.diffusers"),
            # (
            #      "Dockerfile.third_parties",
            #      self.script_dir / "Dockerfile.third_parties"),
            # (
            #     "Dockerfile.nunchaku",
            #     self.script_dir / "Dockerfile.nunchaku")
        ]

def main():
    build_docker = BuildDockerNew()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()
