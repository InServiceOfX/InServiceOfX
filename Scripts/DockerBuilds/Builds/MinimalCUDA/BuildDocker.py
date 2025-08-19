#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationWithNVIDIAGPU,
    BuildDockerImageNoArguments)

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for minimal CUDA",
            Path(__file__).resolve(),
            2,
            configuration_reader_class=ReadBuildConfigurationWithNVIDIAGPU,
            docker_builder_class=BuildDockerImageNoArguments)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.base",
                self.script_dir / "Dockerfile.base"),
            (
                "Dockerfile.LatestKitwareCMake",
                self.parent_dir / "CommonFiles" / "Dockerfile.LatestKitwareCMake"),
            (
                "Dockerfile.rust",
                self.parent_dir / "CommonFiles" / "Dockerfile.rust"),
            (
                "Dockerfile.MoreNvidia",
                self.script_dir / "Dockerfile.MoreNvidia"),
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()