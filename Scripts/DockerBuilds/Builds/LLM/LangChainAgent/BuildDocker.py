#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[3]))
from Utilities import (
    BuildDockerBase,
    ReadBuildConfigurationForMinimalStack,
    BuildDockerImageNoArguments)

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for diffusers with minimal dependencies",
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
                "Dockerfile.minimal_base",
                self.parent_dir / "CommonFiles" / "Dockerfile.minimal_base"),
            (
                "Dockerfile.rust",
                self.parent_dir / "CommonFiles" / "Dockerfile.rust"),
            (
                "Dockerfile.rust",
                self.parent_dir / "CommonFiles" / "Dockerfile.huggingface"),
            (
                "Dockerfile.more_pip_installs",
                self.script_dir / "Dockerfile.langchain"),
            (
                "Dockerfile.apis",
                self.script_dir / "Dockerfile.apis")
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()
