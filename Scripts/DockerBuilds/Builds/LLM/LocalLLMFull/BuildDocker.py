#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[3]))
from Utilities import (
    BuildDockerBaseClass,
    BuildDockerCommand,)
from Utilities.BuildDockerConfiguration import BuildDockerConfiguration

class BuildDocker(BuildDockerBaseClass):
    def __init__(self):

        self._build_directory = Path(__file__).resolve().parent
        build_configuration = BuildDockerConfiguration.load_data(
            self._build_directory / "build_configuration.yml")

        super().__init__(
            "Build Docker image with TensorRT-LLM base",
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
                "Dockerfile.rust",
                self._parent_dir / "CommonFiles" / "Dockerfile.rust"),
            (
                "Dockerfile.huggingface",
                self._parent_dir / "CommonFiles" / "Dockerfile.huggingface"),
            (
                "Dockerfile.more_pip_installs",
                self._build_directory / "Dockerfile.more_pip_installs"),
            (
                "Dockerfile.mcp",
                self.script_dir / "Dockerfile.mcp"),
            # (
            #     "Dockerfile.third_parties",
            #     self.script_dir / "Dockerfile.third_parties"),
            # (
            #     "Dockerfile.langchain",
            #     self.script_dir / "Dockerfile.langchain")
        ]

def main():

    build_docker = BuildDocker()
    parser = build_docker.create_parser()
    args = parser.parse_args()
    build_docker.build(args)

if __name__ == "__main__":
    main()
