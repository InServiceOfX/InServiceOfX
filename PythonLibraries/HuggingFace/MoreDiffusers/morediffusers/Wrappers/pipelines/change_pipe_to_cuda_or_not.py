import torch

def change_pipe_to_cuda_or_not(configuration, pipe):
    if (configuration.is_enable_cpu_offload == False and \
        configuration.is_enable_sequential_cpu_offload == False and \
        configuration.is_to_cuda == True):
        pipe.to("cuda")
