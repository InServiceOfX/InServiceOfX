from corecode.FileIO import get_project_directory_path
from pathlib import Path
import torch
import yaml

class Configuration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreTransformers" / "configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.task = data["task"]
        self.model_path = data["model_path"]
        
        # Extract model name from the last part of the path
        self.model_name = Path(self.model_path).name
        
        self.torch_dtype = data["torch_dtype"]

        if self.torch_dtype == "torch.float16":
            self.torch_dtype = torch.float16
        elif self.torch_dtype == "torch.bfloat16":
            self.torch_dtype = torch.bfloat16
