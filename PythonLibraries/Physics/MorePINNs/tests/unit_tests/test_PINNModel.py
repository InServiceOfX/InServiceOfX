import torch
import numpy as np
from morepinns.PINNModel import PINNDataSet, LearnedTanh, PINNModel

# Test PINNDataSet
def test_pinndataset_initialization():
    data = torch.randn(100, 1)
    label = torch.randn(100, 1)
    dataset = PINNDataSet(data, label)
    
    assert dataset.data.shape == (100, 1)
    assert dataset.label.shape == (100, 1)
    assert len(dataset) == 100

def test_pinndataset_getitem():
    data = torch.randn(100, 1)
    label = torch.randn(100, 1)
    dataset = PINNDataSet(data, label)
    
    item_data, item_label = dataset[0]
    assert item_data.shape == (1,)
    assert item_label.shape == (1,)
    assert torch.equal(item_data, data[0])
    assert torch.equal(item_label, label[0])

# Test LearnedTanh
def test_learned_tanh_initialization():
    activation = LearnedTanh(slope=2.0, n=3.0)
    activation.initialize_with_device(device='cpu', n=3.0)
    assert isinstance(activation.slope, torch.nn.Parameter)
    assert activation.slope.item() == 2.0
    assert activation.n.item() == 3.0

def test_learned_tanh_forward():
    activation = LearnedTanh(slope=1.0, n=1.0)
    input_tensor = torch.tensor([0.0, 1.0, -1.0])
    output = activation(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.all(torch.isfinite(output))  # Check for any NaN or inf values

# Test PINNModel
def test_pinn_model_initialization():
    y0 = torch.zeros((1, 2))  # Example initial condition
    model = PINNModel(nodes=40, layers=2, y0=y0, w_scale=1.0, x_scale=1.0)
    
    # Check model structure
    assert isinstance(model.activation, LearnedTanh)
    assert len(model.seq) == 7  # 3 linear layers + 3 activations + final linear layer
    assert isinstance(model.seq[0], torch.nn.Linear)
    assert model.seq[0].in_features == 1
    assert model.seq[0].out_features == 40

def test_pinn_model_xavier_init():
    y0 = torch.zeros((1, 2))
    model = PINNModel(nodes=40, layers=2, y0=y0)
    model.xavier_init()
    
    # Check if weights are initialized (not zero)
    for m in model.seq:
        if isinstance(m, torch.nn.Linear):
            assert not torch.allclose(m.weight, torch.zeros_like(m.weight))

def test_pinn_model_constant_init():
    y0 = torch.zeros((1, 2))
    model = PINNModel(nodes=40, layers=2, y0=y0)
    w0 = 0.5
    model.constant_init(w0)
    
    # Check if weights are initialized to constant
    for m in model.seq:
        if isinstance(m, torch.nn.Linear):
            assert torch.allclose(m.weight, torch.full_like(m.weight, w0))
            assert torch.allclose(m.bias, torch.full_like(m.bias, w0))

def test_pinn_model_forward():
    y0 = torch.zeros((1, 2))
    model = PINNModel(nodes=40, layers=2, y0=y0, w_scale=1.0, x_scale=1.0)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    
    output = model(x)
    assert output.shape == (3, 2)  # Batch size of 3, output dimension of 2
    assert torch.all(torch.isfinite(output))  # Check for any NaN or inf values

def test_pinn_model_get_slope():
    y0 = torch.zeros((1, 2))
    model = PINNModel(nodes=40, layers=2, y0=y0)
    slope = model.get_slope()
    assert isinstance(slope, float)
    assert slope == model.activation.slope.item() * model.activation.n.item()