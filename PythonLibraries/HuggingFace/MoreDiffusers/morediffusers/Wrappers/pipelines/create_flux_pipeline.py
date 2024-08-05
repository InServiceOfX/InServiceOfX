from diffusers import FluxPipeline

def create_flux_pipeline(
    diffusion_model_subdirectory,
    torch_dtype=None,
    variant=None,
    use_safetensors=None,
    is_enable_cpu_offload=True,
    is_enable_sequential_cpu_offload=True
    ):
    """
    @param variant (`str`, *optional*):
        Load weights from a specified variant filename such as `"fp16"` or
        `"ema"`. This is ignored when loading `from_flax`.

    @details FluxPipeline inherits from DiffusionPipeline and DiffusionPipeline
    defines .from_pretrained(..)
    """

    if variant == None:

        if torch_dtype==None:
            # pipelines/animateddiff/pipeline_animatediff_sdxl.py
            # implements AnimateDiffSDXLPipeline.
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = FluxPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safetensors)
        else:
            pipe = FluxPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safetensors)

    else:

        if torch_dtype==None:
            # pipelines/animateddiff/pipeline_animatediff_sdxl.py
            # implements AnimateDiffSDXLPipeline.
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = FluxPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safetensors,
                variant=variant)
        else:
            pipe = FluxPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
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
