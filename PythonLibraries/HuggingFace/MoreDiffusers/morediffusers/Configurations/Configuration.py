from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class Configuration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.diffusion_model_path = data["diffusion_model_path"]
        self.single_file_diffusion_checkpoint = data[
            "single_file_diffusion_checkpoint"]
        self.temporary_save_path = data["temporary_save_path"]
        self.scheduler = data["scheduler"]
        self.height = data["height"]
        if self.height != None:
            self.height = int(self.height)
        self.width = data["width"]
        if self.width != None:
            self.width = int(self.width)
        self.denoising_end = data["denoising_end"]
        self.guidance_scale = data["guidance_scale"]
        self.clip_skip = data["clip_skip"]
        self.seed = data["seed"]

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