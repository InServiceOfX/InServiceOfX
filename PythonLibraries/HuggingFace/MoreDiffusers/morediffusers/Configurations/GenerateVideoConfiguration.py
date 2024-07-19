from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class GenerateVideoConfiguration:
    """
    @brief Input parameters for running __call__(..)
    """
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "generate_video_configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.image_path = data["image_path"]

        # Default values from pipeline_stable_video_diffusion.py def __call__(..)
        # Optional[int] type.
        self.height = data["height"]
        self.width = data["width"]

        # num_frames: Optional[int] = None
        # Number of video frames to generate. Defaults to
        # `self.unet.config.num_frames` (
        # 14 for `stable-video-diffusion-img2vid`)
        self.num_frames = data["num_frames"]

        self.num_inference_steps = data["num_inference_steps"]
        if self.num_inference_steps != None:
            self.num_inference_steps = int(self.num_inference_steps)
        else:
            self.num_inference_steps = 25

        # Default values from pipeline_stable_video_diffusion.py def __call__(..)
        self.min_guidance_scale = data["min_guidance_scale"]
        if self.min_guidance_scale != None:
            self.min_guidance_scale = float(self.min_guidance_scale)
        else:
            self.min_guidance_scale = 1.0
        self.max_guidance_scale = data["max_guidance_scale"]
        if self.max_guidance_scale != None:
            self.max_guidance_scale = float(self.max_guidance_scale)
        else:
            self.max_guidance_scale = 3.0

        self.fps = data["fps"]
        if self.fps != None:
            self.fps = int(self.fps)
        else:
            self.fps = 7

        # motion_bucket_id ('int', optional, defaults to 127, int = 127)
        # Used for conditioning amount of motion for generation. Higher the
        # number the more motion will be in the video.
        self.motion_bucket_id = data["motion_bucket_id"]
        if self.motion_bucket_id != None:
            self.motion_bucket_id = int(self.motion_bucket_id)
        else:
            self.motion_bucket_id = 127

        # noise_aug_strength (float, optional, defaults to 127)
        # noise_aug_strength: float = 0.02
        # Amount of noise added to init image, higher it is less the video will
        # look like init image. Increase it for more motion.
        self.noise_aug_strength = data["noise_aug_strength"]
        if self.noise_aug_strength != None:
            self.noise_aug_strength = float(self.noise_aug_strength)
        else:
            self.noise_aug_strength = 0.02

        self.decode_chunk_size = data["decode_chunk_size"]

        self.num_videos_per_prompt = data["num_videos_per_prompt"]
        if self.num_videos_per_prompt != None:
            self.num_videos_per_prompt = int(self.num_videos_per_prompt)
        else:
            self.num_videos_per_prompt = 1

        self.seed = data["seed"]

        self.temporary_save_path = data["temporary_save_path"]

    def check_if_paths_exist():
        if (not Path(self.image_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.diffusion_model_path)

        return True