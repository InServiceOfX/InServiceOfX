from diffusers.pipelines import StableDiffusionXLPipeline

def create_stable_diffusion_xl_pipeline(configuration):
    """ 
    Returns:
        Configured StableDiffusionXLPipeline instance

    From
    src/diffusers/pipelines/pipeline_utils.py
    def enable_model_cpu_offload(..):
    Offloads all models to CPU using accelerate, reducing memory usage with low
    impact on performance. Compared to `enable_sequential_cpu_offload`, this
    method moves one whole model at a time to GPU when its `forward` method is
    called, and model remains in GPU until next model runs. Memory savings are
    lower than with `enable_sequential_cpu_offload`, but performance is much
    better due to iterative execution of unet.

    def enable_sequential_cpu_offload(..):
    Offloads all models to CPU using accelerate, significantly reducing memory
    usage. When called, the state dicts of all `torch.nn.Module` components
    (except those in `self._exclude_from_cpu_offload`) are saved to CPU and then
    moved to `torch.device('meta')` and loaded to GPU only when their specific
    submodule has its `forward` method called.
    """
    # Get pretrained kwargs from configuration
    kwargs = configuration.get_pretrained_kwargs()
    kwargs.update({
        "local_files_only": True,
        "add_watermarker": False,
        "use_safetensors": True
    })
    
    # Create pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(configuration.diffusion_model_path),
        **kwargs
    )

    # Configure CPU offloading if enabled
    if configuration.is_enable_model_cpu_offload:
        kwargs = {}
        if getattr(configuration, "cuda_device", None) is not None:
            kwargs["device"] = configuration.cuda_device
        if configuration.get_cuda_device_index() is not None:
            kwargs["gpu_id"] = configuration.get_cuda_device_index()
        pipe.enable_model_cpu_offload(**kwargs)

    if configuration.is_enable_sequential_cpu_offload:
        kwargs = {}
        if getattr(configuration, "cuda_device", None) is not None:
            kwargs["device"] = configuration.cuda_device
        if configuration.get_cuda_device_index() is not None:
            kwargs["gpu_id"] = configuration.get_cuda_device_index()
        pipe.enable_sequential_cpu_offload(**kwargs)

    return pipe
