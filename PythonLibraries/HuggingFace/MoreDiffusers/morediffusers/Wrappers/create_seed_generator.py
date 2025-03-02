import torch

def create_seed_generator(configuration, generation_configuration):

    generator = torch.Generator(device=configuration.cuda_device)
    generator.manual_seed(generation_configuration.seed)
    return generator
