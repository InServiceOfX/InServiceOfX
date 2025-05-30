#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple
sys.path.append(str(Path(__file__).resolve().parents[3]))

from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationWithOpenCV,
    BuildDockerImageWithOpenCV)

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
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()
