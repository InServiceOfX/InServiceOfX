from corecode.FileIO import get_project_directory_path
from pathlib import Path
import torch
import yaml

class VideoConfiguration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "video_configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.diffusion_model_path = data["diffusion_model_path"]
        self.scheduler = data["scheduler"]

        self.torch_dtype = data["torch_dtype"]
        if self.torch_dtype == "torch.float16":
            self.torch_dtype = torch.float16

        self.is_enable_cpu_offload = data["is_enable_cpu_offload"]
        self.is_enable_sequential_cpu_offload = data[
            "is_enable_sequential_cpu_offload"]

        self.is_to_cuda = data["is_to_cuda"]
        self.variant = data["variant"]
        self.use_safetensors = data["use_safetensors"]

    def check_if_paths_exist():
        if (not Path(self.diffusion_model_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.diffusion_model_path)
        elif (not Path(self.temporary_save_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.temporary_save_path)

        return True