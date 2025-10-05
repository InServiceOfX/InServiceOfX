#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple
sys.path.append(str(Path(__file__).resolve().parents[3]))

from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationWithNunchaku,
    BuildDockerImageWithNunchaku)

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
            (
                "Dockerfile.copy_opencv_script",
                self.script_dir / "Dockerfile.copy_opencv_script"),
            (
                "Dockerfile.opencv_with_cuda",
                self.parent_dir / "CommonFiles" / "Dockerfile.opencv_with_cuda"),
            (
                "Dockerfile.diffusers",
                self.script_dir / "Dockerfile.diffusers"),
            (
                 "Dockerfile.third_parties",
                 self.script_dir / "Dockerfile.third_parties"),
            (
                "Dockerfile.nunchaku",
                self.script_dir / "Dockerfile.nunchaku")
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()
