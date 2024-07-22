import torch

def change_video_pipe_to_cuda_or_not(configuration, pipe):
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