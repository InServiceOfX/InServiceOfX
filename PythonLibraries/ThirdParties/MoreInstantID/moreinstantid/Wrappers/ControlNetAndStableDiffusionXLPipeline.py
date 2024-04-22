from diffusers.models import ControlNetModel
from moreinsightface.Wrappers import FaceAnalysisWrapper
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

import diffusers

class ControlNetAndStableDiffusionXLPipeline:
	"""
	@details

	After construction of the class instance with __init__(..), the steps before
	.generate_image(..) alluded to in app.py of InstantID are as follows:

	* pipe.load_ip_adapter_instantid(face_adapter)
	* load load weights and then disable_lora() (EY: ??)

	Within generate_image(..) in app.py of InstantID, they first do these steps:
	* enable_LCM or not, where the pipe.scheduler is set.
	* Extract face and pose information (if pose image is given) from images,
	which then gets fed into the pipeline.__call__(..) function.
	* optionally create a control_mask.
	"""
	def __init__(
		self,
		controlnet_path,
		diffusion_model_subdir,
		torch_dtype=None,
		is_enable_cpu_offload=True,
		is_enable_sequential_cpu=True,
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

		if (is_enable_cpu_offload):
			self._offload_to_cpu()

		if (is_enable_cpu_offload and is_enable_sequential_cpu):
			self._enable_sequential_cpu_offload()

	def _offload_to_cpu(self):
		self.pipe.enable_model_cpu_offload()

	def _enable_sequential_cpu_offload(self):
		"""
		@details When running these models on older and less capable GPUs, I
		found this step to be critical, important, a necessary step to run
		calling (i.e. invoking .__call__()) the pipe,
		StableDiffusionXLInstantIDPipeline.

		NOTE: It's worth running this again right before you run generate image,
		to follow what's done in the CPU offloading example here:
		https://huggingface.co/docs/diffusers/en/optimization/memory
		"""
		self.pipe.enable_sequential_cpu_offload()

	def load_ip_adapter(self,
		ip_adapter_path,
		ip_adapter_image_embedding_dimension=512,
		ip_adapter_number_of_tokens=16,
		ip_adapter_scale=0.5
		):
		"""
		@param ip_adapter_image_embedding_dimension=512 This was a parameter
		into load_ip_adapter_instantid but its default value was used in
		InstantID's application.
		@param ip_adapter_number_of_tokens=16 This was a parameter into
		load_ip_adapter_instantid but its default value was used in InstantID's
		application.

		@details This needs to be run once before generating the first image.
		"""
		self.pipe.load_ip_adapter_instantid(
			ip_adapter_path,
			ip_adapter_image_embedding_dimension,
			ip_adapter_number_of_tokens,
			ip_adapter_scale)

	def _use_euler_discrete_scheduler(self):
		"""
		@details Optionally, run this before generate_image(..).
		"""
		self.pipe.disable_lora()
		self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
			self.pipe.scheduler.config)

	def generate_image(
		self,
		prompt,
		face_information,
		ip_adapter_scale=1.0,
		pose_information=None,
		negative_prompt=None,
		controlnet_conditioning_scale=1.0,
		number_of_steps=50,
		guidance_scale=None
		):
		"""
		@param controlnet_conditioning_scale In the app(lication) from InstantID
		it's called identitynet_strength_ratio with label "IdentityNet strength
		(for fidelity)". But literally, in
		pipeline_stable_diffusion_xl_instantid of InstantID, the outputs of
		ControlNet are multiplied by this scale, before added to residual in
		original unet. StableDiffusionXLInstantIDPipeline defaults this to 1.0,
		the app(lication) of InstantID defaults to 0.8

		@param ip_adapter_scale In pipeline_stable_diffusion_xl_instantid of
		InstantID, this literally sets a data member, .scale, to this value, for
		an instance of IPAttnProcessor. In app(lication) of InstantID, it's
		called "adapter_strength_ratio", and suggested values are from 0 to 1.5,
		and default 0.8.
		"""
		keypoints = None
		width = None
		height = None
		if pose_information == None:
			keypoints = face_information.face_keypoints
			height = face_information.height
			width = face_information.width
		else:
			keypoints = pose_information.pose_keypoints
			height = pose_information.height
			width = pose_information.width

		if negative_prompt == None:
			negative_prompt = ""

		self.pipe.set_ip_adapter_scale(ip_adapter_scale)

		if guidance_scale == None:
			images = self.pipe(
				prompt=prompt,
				negative_prompt=negative_prompt,
				image_embeds=face_information.face_embedding,
				image=keypoints,
				controlnet_conditioning_scale=float(controlnet_conditioning_scale),
				num_inference_steps=number_of_steps,
				height=height,
				width=width)
		else:
			images = self.pipe(
				prompt=prompt,
				negative_prompt=negative_prompt,
				image_embeds=face_information.face_embedding,
				image=keypoints,
				controlnet_conditioning_scale=float(controlnet_conditioning_scale),
				num_inference_steps=number_of_steps,
				guidance_scale=float(guidance_scale),
				height=height,
				width=width)
		return images[0]
