#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[3]))
from Utilities import (
    BuildDockerBase,
    BuildDockerImageNoArguments,
    ReadBuildConfigurationForMinimalStack)

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            (
                "Build Docker image for a full stack application / website "
                "with minimal setup."),
            Path(__file__).resolve(),
            3,
            configuration_reader_class=ReadBuildConfigurationForMinimalStack,
            docker_builder_class=BuildDockerImageNoArguments)

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:

        return [
            (
                "Dockerfile.header",
                self.parent_dir / "CommonFiles" / "Dockerfile.header"),
            (
                "Dockerfile.base",
                self.parent_dir / "CommonFiles" / "Dockerfile.base"),
            (
                "Dockerfile.nvm_latest",
                self.script_dir / "Dockerfile.nvm_latest"),
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()
