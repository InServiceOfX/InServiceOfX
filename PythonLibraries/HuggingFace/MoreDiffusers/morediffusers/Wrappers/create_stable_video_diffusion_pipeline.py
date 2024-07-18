from diffusers.pipelines import StableVideoDiffusionPipeline

import torch

def create_stable_video_diffusion_pipeline(
    diffusion_model_subdirectory,
    torch_dtype=None,
    variant=None,
    use_safe_tensors=None,
    is_enable_cpu_offload=True,
    is_enable_sequential_cpu=True
    ):
    """
    pipeline_utils.py, def from_pretrained(..)
    """
    if variant == None:

        if torch_dtype==None:
            # pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
            # implements StableDiffusionXLPipeline
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safe_tensors)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safe_tensors)

    else:

        if torch_dtype==None:
            # pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
            # implements StableDiffusionXLPipeline
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                use_safetensors=use_safe_tensors,
                variant=variant)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                use_safetensors=use_safe_tensors,
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


def change_pipe_to_cuda_or_not(configuration, pipe):
    """
    https://huggingface.co/docs/diffusers/en/using-diffusers/text-img2vid#optimize
    """
    if (configuration.is_enable_cpu_offload == False and \
        configuration.is_enable_sequential_cpu_offload == False and \
        configuration.is_to_cuda == True):
        pipe.to("cuda")

        pipe.unet = torch.compile(
            pipe.unet,
            mode="reduce-overhead",
            fullgraph=True)