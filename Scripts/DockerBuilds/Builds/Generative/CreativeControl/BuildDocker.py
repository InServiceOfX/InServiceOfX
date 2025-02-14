#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
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
            # Count how many, from 0, parents to get to the Utilities directory.
            3,
            configuration_reader_class=ReadBuildConfigurationWithOpenCV,
            docker_builder_class=BuildDockerImageWithOpenCV)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.base",
                self.parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.rust",
                self.parent_dir / "CommonFiles" / "Dockerfile.rust"),
            (
                "Dockerfile.opencv_with_cuda",
                self.parent_dir / "CommonFiles" / "Dockerfile.opencv_with_cuda"),
            (
                "Dockerfile.lumaai",
                self.script_dir / "Dockerfile.lumaai"),
            (
                "Dockerfile.falai",
                self.script_dir / "Dockerfile.falai")
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    build_docker = BuildDocker()
    build_docker.build(args)

    print(
        f"Successfully built Docker image '{build_docker.configuration['DOCKER_IMAGE_NAME']}'.")

if __name__ == "__main__":
    main()
