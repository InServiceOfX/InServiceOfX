from diffusers import AnimateDiffSDXLPipeline

def create_animate_diff_sdxl_pipeline(
    diffusion_model_subdirectory,
    motion_adapter,
    torch_dtype=None,
    variant=None,
    use_safetensors=None,
    is_enable_cpu_offload=True,
    is_enable_sequential_cpu_offload=True
    ):
    """
    See pipeline_utils.py in diffusers for def from_pretrained(..) in
    DiffusionPipeline class for the 
    """

    if variant == None:

        if torch_dtype==None:
            # pipelines/animateddiff/pipeline_animatediff_sdxl.py
            # implements AnimateDiffSDXLPipeline.
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                motion_adapter=motion_adapter,
                local_files_only=True,
                use_safetensors=use_safetensors)
        else:
            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                motion_adapter=motion_adapter,
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safetensors)

    else:

        if torch_dtype==None:
            # pipelines/animateddiff/pipeline_animatediff_sdxl.py
            # implements AnimateDiffSDXLPipeline.
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                motion_adapter=motion_adapter,
                local_files_only=True,
                use_safetensors=use_safetensors,
                variant=variant)
        else:
            pipe = AnimateDiffSDXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                motion_adapter=motion_adapter,
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safetensors,
                variant=variant)

    if (is_enable_cpu_offload):
        pipe.enable_model_cpu_offload()

    if (is_enable_cpu_offload and is_enable_sequential_cpu_offload):
        """
        NOTE: It's worth running this again right before you run generate image,
        to follow what's done in the CPU offloading example here:
        https://huggingface.co/docs/diffusers/en/optimization/memory
        """
        pipe.enable_sequential_cpu_offload()

    return pipe

