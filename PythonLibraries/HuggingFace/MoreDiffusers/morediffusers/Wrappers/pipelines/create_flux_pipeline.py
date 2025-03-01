from diffusers import FluxPipeline
from typing import Optional, Dict, Any

def create_flux_pipeline(configuration):
    kwargs = configuration.get_pretrained_kwargs()
    kwargs.update({
        "local_files_only": True,
        "use_safetensors": True
    })

    pipe = FluxPipeline.from_pretrained(
        str(configuration.diffusion_model_path),
        **kwargs
    )

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
