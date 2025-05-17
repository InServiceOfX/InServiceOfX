from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

def create_stable_diffusion_xl_pipeline(
	controlnet,
	diffusion_model_subdirectory,
	ip_adapter_path,
	ip_adapter_image_embedding_dimension=512,
	ip_adapter_number_of_tokens=16,
	ip_adapter_scale=0.5,
	torch_dtype=None,
	is_enable_cpu_offload=True,
	is_enable_sequential_cpu=True
	):
	"""
	@param ip_adapter_image_embedding_dimension=512 This was a parameter
	into load_ip_adapter_instantid but its default value was used in
	InstantID's application.
	@param ip_adapter_number_of_tokens=16 This was a parameter into
	load_ip_adapter_instantid but its default value was used in InstantID's
	application.

	@details 
	After creating an instance of StableDiffusionXLInstantIDPipeline, the steps
	before generate_image(..) alluded to in app.py of InstantID are as follows:

	* pipe.load_ip_adapter_instantid(face_adapter)
	* load load weights and then disable_lora() (EY: ??)
	  - I found that we're missing some backend, 
	   ValueError: PEFT backend is required for this method.

	Within generate_image(..) in app.py of InstantID, they first do these steps:
	* enable_LCM or not, where the pipe.scheduler is set.
	* Extract face and pose information (if pose image is given) from images,
	which then gets fed into the pipeline.__call__(..) function.
	* optionally create a control_mask.
	"""
	if torch_dtype==None:
        # pipelines/controlnet/pipeline_controlnet_sd_xl.py implements
        # StableDiffusionXLControlNetPipeline
        # from_pretrained(..) defined in DiffusionPipeline in
        # diffusers/src/diffusers/pipelines/pipeline_utils.py
		pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
			str(diffusion_model_subdirectory),
			controlnet=controlnet,
			local_files_only=True,
			feature_extractor=None)
	else:
		pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
			str(diffusion_model_subdirectory),
			controlnet=controlnet,
			torch_dtype=torch_dtype,
			local_files_only=True,
			feature_extractor=None)

	if (is_enable_cpu_offload):
		pipe.enable_model_cpu_offload()

	# If this is run in a class and *not* in the init, then running this
	# separately, but as class member functions, would result in this error:
	#   warnings.warn(f'for {key}: copying from a non-meta parameter in the checkpoint to a meta '
#/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py:2024: UserWarning: for 139.to_v_ip.weight: copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op. (Did you mean to pass `assign=True` to assign items in the state dictionary to their corresponding key in the module instead of copying them in place?)
#  warnings.warn(f'for {key}: copying from a non-meta parameter in the checkpoint to a meta '
	pipe.load_ip_adapter_instantid(
		ip_adapter_path,
		ip_adapter_image_embedding_dimension,
		ip_adapter_number_of_tokens,
		ip_adapter_scale)

	if (is_enable_cpu_offload and is_enable_cpu_offload):
		"""
		When running these models on older and less capable GPUs, I found this
		step to be critical, important, a necessary step to run calling (i.e.
		invoking .__call__()) the pipe, StableDiffusionXLInstantIDPipeline.

		NOTE: It's worth running this again right before you run generate image,
		to follow what's done in the CPU offloading example here:
		https://huggingface.co/docs/diffusers/en/optimization/memory
		"""
		pipe.enable_sequential_cpu_offload()

	return pipe