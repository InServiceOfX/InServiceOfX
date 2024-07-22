from diffusers.pipelines import I2VGenXLPipeline

import torch

def create_i2vgen_xl_pipeline(
    diffusion_model_subdirectory,
    torch_dtype=None,
    variant=None,
    use_safetensors=None,
    is_enable_cpu_offload=True,
    is_enable_sequential_cpu=True
    ):
    """
    pipeline_utils.py, def from_pretrained(..)
    """
    if variant == None:

        if torch_dtype==None:
            # pipelines/i2vgen_xl/pipeline_i2vgen_xl.py
            # implements I2VGenXLPipeline
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = I2VGenXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safetensors)
        else:
            pipe = I2VGenXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safetensors)

    else:

        if torch_dtype==None:
            # pipelines/i2vgen_xl/pipeline_i2vgen_xl.py
            # implements I2VGenXLPipeline
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = I2VGenXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safetensors,
                variant=variant)
        else:
            pipe = I2VGenXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safetensors,
                variant=variant)

    if (is_enable_cpu_offload):
        pipe.enable_model_cpu_offload()

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
