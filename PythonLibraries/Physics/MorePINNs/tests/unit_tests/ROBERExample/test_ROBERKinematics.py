import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from morepinns.ROBERExample.ROBERKinetics import ROBERModel

class TestROBERProblem:
    @pytest.fixture
    def setup(self):
        """Setup fixture for tests"""
        model = ROBERModel()
        return model

    def test_model_initialization(self, setup):
        """Test model initialization"""
        model = setup
        assert isinstance(model, ROBERModel)
        assert model.pde_fn.k1 == 0.04
        assert model.pde_fn.k2 == 3e7
        assert model.pde_fn.k3 == 1e4

    def test_model_output_shape(self, setup):
        """Test model output shape"""
        model = setup
        t = torch.logspace(-6, 4, 100).reshape(-1, 1)
        output = model(time=t)  # Direct call
        assert output.shape == (100, 3)

    def test_pde_residuals(self, setup):
        """Test PDE residual computation"""
        model = setup
        t = torch.logspace(-6, 4, 100).reshape(-1, 1)
        t.requires_grad = True
        y = model(t)  # Direct call
        residuals = model.pde_fn(t, y)
        assert residuals.shape == (100, 3)

    def test_conservation_of_mass(self, setup):
        """Test conservation of mass"""
        model = setup
        t = torch.logspace(-6, 4, 100).reshape(-1, 1)
        with torch.no_grad():
            predictions = model(t)  # Direct call
            total_concentration = torch.sum(predictions, dim=1)
 