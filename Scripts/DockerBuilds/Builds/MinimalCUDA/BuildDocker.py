#!/usr/bin/env python3

import sys
from pathlib import Path

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationForNVIDIAGPU,
    BuildDockerImageNoArguments)

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for minimal CUDA",
            Path(__file__).resolve(),
            2,
            configuration_reader_class=ReadBuildConfigurationForNVIDIAGPU,
            docker_builder_class=BuildDockerImageNoArguments)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.rust",
                self.parent_dir / "CommonFiles" / "Dockerfile.rust"),
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()

if __name__ == "__main__":
    main()