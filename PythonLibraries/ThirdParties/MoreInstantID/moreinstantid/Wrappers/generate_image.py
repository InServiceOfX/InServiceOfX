def generate_image(
	pipe,
	prompt,
	face_information,
	negative_prompt=None,
	pose_information=None,
	ip_adapter_scale=1.0,
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

	pipe.set_ip_adapter_scale(ip_adapter_scale)

	if guidance_scale == None:
		# __call__() will return a type
		# 'diffusers.pipelines.stable_diffusion_xl.pipeline_output.StableDiffusionXLPipelineOutput'
		# and can be index by number, yielding a list.
		# The class member .images is also a list, a list of PIL.Image.Image.
		images = pipe(
			prompt=prompt,
			negative_prompt=negative_prompt,
			image_embeds=face_information.face_embedding,
			image=keypoints,
			controlnet_conditioning_scale=float(controlnet_conditioning_scale),
			num_inference_steps=number_of_steps,
			height=height,
			width=width).images
	else:
		images = pipe(
			prompt=prompt,
			negative_prompt=negative_prompt,
			image_embeds=face_information.face_embedding,
			image=keypoints,
			controlnet_conditioning_scale=float(controlnet_conditioning_scale),
			num_inference_steps=number_of_steps,
			guidance_scale=float(guidance_scale),
			height=height,
			width=width).images

	return images[0]