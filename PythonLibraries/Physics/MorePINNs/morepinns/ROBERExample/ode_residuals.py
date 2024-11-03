
def ode_residuals(model, t):
    y = model(t)
    y1 = y[:, 0].view(-1, 1)
    y2 = y[:, 1].view(-1, 1)
    y3 = y[:, 2].view(-1, 1)

    # Ensure gradients can be computed
    y1.requires_grad_(True)
    y2.requires_grad_(True)
    y3.requires_grad_(True)
    
    # Compute derivatives
    dy1_dt = torch.autograd.grad(y1, t, grad_outputs=torch.ones_like(y1), create_graph=True)[0]
    dy2_dt = torch.autograd.grad(y2, t, grad_outputs=torch.ones_like(y2), create_graph=True)[0]
    dy3_dt = torch.autograd.grad(y3, t, grad_outputs=torch.ones_like(y3), create_graph=True)[0]
    
    # Reaction rate constants
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    
    # ODE residuals
    res1 = dy1_dt + k1 * y1 - k3 * y2 * y3
    res2 = dy2_dt - k1 * y1 + k2 * y2 ** 2 + k3 * y2 * y3
    res3 = dy3_dt - k2 * y2 ** 2
    
    return res1, res2, res3