from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

import diffusers

class ControlNetAndStableDiffusionXLPipeline:
	def __init__(
		self,
		controlnet_path,
		diffusion_model_subdir
		torch_dtype=None
		):
		if torch_dtype==None:
			self.control_net = ControlNetModel.from_pretrained(
				str(controlnet_path))
			self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
				str(diffusion_model_subdir),
				controlnet=self.control_net,
				local_files_only=True,
				feature_extractor=None)
		else:
			self.control_net = ControlNetModel.from_pretrained(
				str(controlnet_path),
				torch_dtype=torch_dtype)
			self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
				str(diffusion_model_subdir),
				controlnet=self.control_net,
				torch_dtype=torch_dtype,
				local_files_only=True,
				feature_extractor=None)

	def use_euler_discrete_scheduler(self):
		self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
			self.pipe.scheduler.config)

	def offload_to_cpu(self):
		self.pipe.enable_model_cpu_offload()
