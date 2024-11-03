import torch
from pinnstorch.models import FCN, PINNModule
import numpy as np
import matplotlib.pyplot as plt

from typing import List

class MethaneLOXKinetics:
    def __init__(self):
        # Further reduce rate constants and use log scale
        self.k1 = 0.04     # log(7.9e13) normalized
        self.k2 = 0.03     # log(2.31e12) normalized
        self.k3 = 0.02     # log(3.43e9) normalized
        self.k4 = 0.04     # log(1.87e17) normalized
        self.k5 = 0.01     # log(4.76e7) normalized


def create_training_data(n_points: int = 1000):
    """Create time points for training"""
    t = np.linspace(0, 1, n_points)
    # Create a constant spatial point
    x = np.zeros_like(t)
    t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    t.requires_grad = True
    return x, t


class CombustionModel(PINNModule):
    def __init__(self):
        layers = [2, 64, 128, 128, 64, 10]
        lb = torch.tensor([0.0, 0.0], dtype=torch.float32)
        ub = torch.tensor([0.0, 1.0], dtype=torch.float32)
        output_names = ['CH4', 'O2', 'CH3', 'HO2', 'CH2O', 'OH', 'HCO', 'H2O', 'CO', 'CO2']
        
        net = FCN(
            layers=layers,
            lb=lb,
            ub=ub,
            output_names=output_names
        )
        
        # Initialize weights with smaller values and ensure float32
        for param in net.parameters():
            param.data = param.data.float()  # Convert to float32
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.uniform_(param, -0.1, 0.1)
        
        pde_fn = CombustionPDEFunc()
        super().__init__(net=net, pde_fn=pde_fn)

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        # Ensure float32 and add small noise
        spatial = [s.float() + 1e-8 for s in spatial]  # Convert to float32
        time = time.float() + 1e-8  # Convert to float32
        return self.net.forward(spatial=spatial, time=time)



class CombustionPDEFunc:
    def __init__(self):
        self.kinetics = MethaneLOXKinetics()
    
    def __call__(self, X, y_dict):
        # Ensure float32
        X = X.float()
        t = X[:, 1:2]
        
        species_names = ['CH4', 'O2', 'CH3', 'HO2', 'CH2O', 'OH', 'HCO', 'H2O', 'CO', 'CO2']
        y = torch.stack([torch.nn.functional.softplus(y_dict[name].float()) 
                        for name in species_names], dim=1)
        
        dy_dt = torch.autograd.grad(
            y, t,
            grad_outputs=torch.ones_like(y, dtype=torch.float32),
            create_graph=True,
            allow_unused=True
        )[0]

        if dy_dt is None:
            dy_dt = torch.zeros_like(y, dtype=torch.float32)

        # Ensure all computations are in float32
        r1 = (self.kinetics.k1 * torch.nn.functional.softplus(y[:, 0] * y[:, 1])).float()
        r2 = (self.kinetics.k2 * torch.nn.functional.softplus(y[:, 2] * y[:, 1])).float()
        r3 = (self.kinetics.k3 * torch.nn.functional.softplus(y[:, 4] * y[:, 5])).float()
        r4 = (self.kinetics.k4 * torch.nn.functional.softplus(y[:, 6])).float()
        r5 = (self.kinetics.k5 * torch.nn.functional.softplus(y[:, 8] * y[:, 5])).float()

        scale_factor = torch.tensor(1e-2, dtype=torch.float32)
        equations = [
            scale_factor * (dy_dt[:, 0] + r1).float(),
            scale_factor * (dy_dt[:, 1] + r1 + r2).float(),
            scale_factor * (dy_dt[:, 2] - r1 + r2).float(),
            scale_factor * (dy_dt[:, 3] - r1).float(),
            scale_factor * (dy_dt[:, 4] - r2 + r3).float(),
            scale_factor * (dy_dt[:, 5] - r2 + r3 + r5).float(),
            scale_factor * (dy_dt[:, 6] - r3 + r4).float(),
            scale_factor * (dy_dt[:, 7] - r3).float(),
            scale_factor * (dy_dt[:, 8] - r4 + r5).float(),
            scale_factor * (dy_dt[:, 9] - r5).float()
        ]
        
        return torch.stack(equations, dim=1).float()



def train_model(model: CombustionModel, X_train: tuple, epochs: int = 10000):
    """Train the model with gradient clipping and adjusted learning rate"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate
    
    spatial, time = X_train
    spatial = spatial.detach().clone().requires_grad_(True)
    time = time.detach().clone().requires_grad_(True)
    
    # Initial conditions (t=0)
    ic_value = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                           dtype=torch.float32)
    
    x_ic = torch.tensor([[0.0]], dtype=torch.float32)
    t_ic = torch.tensor([[0.0]], dtype=torch.float32)
    
    species_names = ['CH4', 'O2', 'CH3', 'HO2', 'CH2O', 'OH', 'HCO', 'H2O', 'CO', 'CO2']
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, verbose=True
    )
    
    # Keep track of best loss
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 1000
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        X = torch.cat([spatial, time], dim=1)
        
        try:
            # Get predictions and compute losses
            y_pred_dict = model(spatial=[spatial], time=time)
            residuals = model.pde_fn(X, y_pred_dict)
            pde_loss = torch.mean(residuals**2)
            
            ic_pred_dict = model(spatial=[x_ic], time=t_ic)
            ic_pred = torch.stack([ic_pred_dict[name] for name in species_names], dim=1)
            ic_loss = torch.mean((ic_pred - ic_value)**2)
            
            # Total loss with scaled components
            loss = pde_loss + 10.0 * ic_loss  # Reduced IC loss weight
            
            # Check for NaN
            if torch.isnan(loss):
                print("NaN detected, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > max_patience:
                print("Early stopping triggered")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}, "
                      f"PDE Loss: {pde_loss.item():.6f}, "
                      f"IC Loss: {ic_loss.item():.6f}")
                
        except RuntimeError as e:
            print(f"Error in epoch {epoch}: {e}")
            continue
    
    return model

def plot_results(model: CombustionModel, t_eval: np.ndarray):
    """Plot the results"""
    # Create evaluation points with dummy spatial dimension
    x_eval = torch.zeros_like(torch.tensor(t_eval)).reshape(-1, 1)
    t_eval_tensor = torch.tensor(t_eval, dtype=torch.float32).reshape(-1, 1)
    
    with torch.no_grad():
        predictions = model.forward([x_eval], t_eval_tensor)
        
    species_names = model.net.output_names
    
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(species_names):
        plt.plot(t_eval, predictions[:, i].numpy(), label=name)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    plt.title('Methane-LOX Combustion Kinetics')
    plt.show()