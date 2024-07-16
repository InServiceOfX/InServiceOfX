from diffusers.pipelines import StableDiffusionXLPipeline

def create_stable_diffusion_xl_pipeline(
    diffusion_model_subdirectory,
    single_file_checkpoint_path=None,
    torch_dtype=None,
    is_enable_cpu_offload=True,
    is_enable_sequential_cpu=True
    ):

    if single_file_checkpoint_path == None:

        if torch_dtype==None:
            # pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
            # implements StableDiffusionXLPipeline
            # from_pretrained(..) defined in DiffusionPipeline in
            # diffusers/src/diffusers/pipelines/pipeline_utils.py
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                local_files_only=True,
                add_watermarker=False,
                use_safetensors=True)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(diffusion_model_subdirectory),
                torch_dtype=torch_dtype,
                local_files_only=True,
                add_watermarker=False,
                use_safetensors=True)

    else:

        # TODO: running single file gives this error:
        # OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'openai/clip-vit-large-patch14' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.
        # ValueError: With local_files_only set to True, you must first locally save the text_encoder and tokenizer in the following path: 'openai/clip-vit-large-patch14'.

        if torch_dtype==None:

            pipe = StableDiffusionXLPipeline.from_single_file(
                single_file_checkpoint_path,
                config=diffusion_model_subdirectory,
                local_files_only=True)
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(
                single_file_checkpoint_path,
                torch_dtype=torch_dtype,
                config=diffusion_model_subdirectory,
                local_files_only=True)

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
    if (configuration.is_enable_cpu_offload == False and \
        configuration.is_enable_sequential_cpu_offload == False and \
        configuration.is_to_cuda == True):
        pipe.to("cuda")
