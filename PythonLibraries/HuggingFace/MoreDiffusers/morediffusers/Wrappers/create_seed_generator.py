import torch

def create_seed_generator(configuration, seed_value):

    seed = int(seed_value)

    if configuration.is_enable_cpu_offload == False and \
        configuration.is_enable_sequential_cpu_offload == False:
        # TODO: determine where the generator should be.
        #and \
        #(configuration.is_to_cuda == False or \
        #    configuration.is_to_cuda == None):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    else:
        # https://pytorch.org/docs/stable/generated/torch.Generator.html
        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)
    return generator