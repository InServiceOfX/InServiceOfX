#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List, Tuple

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[3]))
from Utilities import (
    BuildDockerBase,
    ReadBuildConfiguration,
    BuildDockerImage)

class BuildDockerImageWithCUDAARCH(BuildDockerImage):
    def __init__(self):
        super().__init__(["CUDA_ARCH",])

class ReadBuildConfigurationWithCUDAARCH(ReadBuildConfiguration):
    def __init__(self):
        required_keys = {
            "DOCKER_IMAGE_NAME",
            "BASE_IMAGE",
            "CUDA_ARCH"}

        super().__init__(required_keys)

class BuildDocker(BuildDockerBase):
    def __init__(self):
        super().__init__(
            "Build Docker image for diffusers with minimal dependencies",
            Path(__file__).resolve(),
            3,
            configuration_reader_class=ReadBuildConfigurationWithCUDAARCH,
            docker_builder_class=BuildDockerImageWithCUDAARCH)

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
                "Dockerfile.nvidia_tensorrt_llm",
                self.script_dir / "Dockerfile.nvidia_tensorrt_llm"),
            (
                "Dockerfile.huggingface",
                self.parent_dir / "CommonFiles" / "Dockerfile.huggingface"),
            (
                "Dockerfile.more_pip_installs",
                self.script_dir / "Dockerfile.more_pip_installs"),
            ## TODO: See why there are version mismatches when including
            # building with SGLang.
            # (
            #     "Dockerfile.sglang",
            #     self.script_dir / "Dockerfile.sglang"),
            (
                "Dockerfile.mcp",
                self.script_dir / "Dockerfile.mcp"),
            (
                "Dockerfile.third_parties",
                self.script_dir / "Dockerfile.third_parties"),
            (
                "Dockerfile.langchain",
                self.script_dir / "Dockerfile.langchain")
        ]

def main():
    parser = BuildDocker().create_parser()
    args = parser.parse_args()
    BuildDocker().build(args)

if __name__ == "__main__":
    main()
