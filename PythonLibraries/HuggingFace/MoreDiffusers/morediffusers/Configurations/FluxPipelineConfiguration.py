from corecode.FileIO import get_project_directory_path
from pathlib import Path
import torch
import yaml

class FluxPipelineConfiguration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "flux_pipeline_configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.diffusion_model_path = data["diffusion_model_path"]
        self.temporary_save_path = data["temporary_save_path"]
        self.scheduler = data["scheduler"]
        self.a1111_kdiffusion = str(data["A1111_kdiffusion"])
        self.height = data["height"]
        if self.height != None:
            self.height = int(self.height)
        self.width = data["width"]
        if self.width != None:
            self.width = int(self.width)
        self.seed = data["seed"]

        self.max_sequence_length = data["max_sequence_length"]
        if self.max_sequence_length == None:
        	self.max_sequence_length = 512
        else:
        	self.max_sequence_length = int(self.max_sequence_length)

        self.torch_dtype = data["torch_dtype"]

        if self.torch_dtype == "torch.float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "torch.bfloat16":
            self.torch_dtype = torch.bfloat16

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