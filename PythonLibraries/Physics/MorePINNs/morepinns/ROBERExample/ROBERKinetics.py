import torch
from pinnstorch.models import FCN, PINNModule

class ROBERPDEFunc:
    def __init__(self):
        self.k1 = 0.04
        self.k2 = 3e7
        self.k3 = 1e4

    def __call__(self, x, y):
        dy_dt = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        
        eq1 = dy_dt[:, 0] + self.k1 * y[:, 0] - self.k3 * y[:, 1] * y[:, 2]
        eq2 = dy_dt[:, 1] - self.k1 * y[:, 0] + self.k2 * y[:, 1]**2 + self.k3 * y[:, 1] * y[:, 2]
        eq3 = dy_dt[:, 2] - self.k2 * y[:, 1]**2
        
        return torch.stack([eq1, eq2, eq3], dim=1)

class ROBERModel(PINNModule):
    def __init__(self):
        layers = [1, 100, 100, 100, 100, 100, 3]
        lb = [0.0]
        ub = [1e4]
        output_names = ['A', 'B', 'C']
        
        net = FCN(
            layers=layers,
            lb=lb,
            ub=ub,
            output_names=output_names
        )
        pde_fn = ROBERPDEFunc()
        super().__init__(net=net, pde_fn=pde_fn)

    def forward(self, x):
        return self.net(x)