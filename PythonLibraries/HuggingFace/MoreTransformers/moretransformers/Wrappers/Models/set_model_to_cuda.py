import torch

def set_model_to_cuda(model):
    device = torch.device('cuda')

    model.to(device)
