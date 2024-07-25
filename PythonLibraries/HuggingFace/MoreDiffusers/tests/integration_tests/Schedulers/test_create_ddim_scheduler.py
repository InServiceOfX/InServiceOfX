from corecode.Utilities import DataSubdirectories

from morediffusers.Schedulers import create_ddim_scheduler

import pytest
import torch

data_sub_dirs = DataSubdirectories()

def test_create_ddim_scheduler_creates_with_sdxl_model():
    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "dataautogpt3" / "ProteusV0.5"

    scheduler = create_ddim_scheduler(
        pretrained_diffusion_model_name_or_path,
        subfolder="scheduler")

    # Since in scheduling_ddim.py, DDIMScheduler.__init__(..), beta_schedule==
    # "linear", then .betas is set from 0.0001 to 0.02 for beta_start, beta_end
    # default values as input argumnets to torch.linspace
    assert isinstance(scheduler.betas, torch.Tensor)
    assert isinstance(scheduler.alphas, torch.Tensor)
    # TODO: Why isn't expected to be 1.0, from the code for
    # DDIMScheduler.__init__(..)?
    assert scheduler.final_alpha_cumprod.item() == \
        pytest.approx(0.9991499781608582, abs=1e-19)
    assert scheduler.init_noise_sigma == 1.0
    assert scheduler.num_inference_steps == None
    assert isinstance(scheduler.timesteps, torch.Tensor)
    assert scheduler.timesteps.shape[0] == 1000