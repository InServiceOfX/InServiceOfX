from corecode.Utilities import (
    DataSubdirectories,
    )
from morediffusers.Schedulers import change_scheduler_or_not
from moreinstantid.Configuration import Configuration
from moreinstantid.Wrappers import (
    create_controlnet,
    create_stable_diffusion_xl_pipeline
    )
from pathlib import Path
from diffusers import (
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler
    )

import pytest
import torch

test_data_directory = Path(__file__).resolve().parent.parent.parent / "TestData"

def test_changes_scheduler_on_stable_diffusion_xl_pipeline():
    test_file_path = test_data_directory / "integration_configuration_1.yml"
    assert test_file_path.exists()

    configuration = Configuration(test_file_path)

    # The ControlNet binary offered by InstantID doesn't seem to work with
    # 16-bit float type from torch because on CPU, there's no 16-bit float type.
    # So we didn't add the torch_dtype argument.
    controlnet = create_controlnet(configuration.control_net_model_path)
    pipe = create_stable_diffusion_xl_pipeline(
        controlnet,
        configuration.diffusion_model_path,
        configuration.ip_adapter_path,
        # On notebook NVIDIA RTX 3070, this process gets Killed, possibly from
        # overheating, if we try to load the entire model onto the GPU.
        #torch_dtype=torch.float16,
        is_enable_cpu_offload=True,
        is_enable_sequential_cpu=True)

    assert isinstance(pipe.scheduler, EulerAncestralDiscreteScheduler)
    assert pipe.scheduler.config._class_name == \
        "EulerAncestralDiscreteScheduler"
    assert pipe.scheduler.config._diffusers_version == "0.22.0.dev0"
    assert pipe.scheduler.config.beta_end == 0.012
    assert pipe.scheduler.config.beta_schedule == "scaled_linear"
    assert pipe.scheduler.config.beta_start == 0.00085
    assert pipe.scheduler.config.clip_sample == False
    assert pipe.scheduler.config.interpolation_type == "linear"
    assert pipe.scheduler.config.num_train_timesteps == 1000
    assert pipe.scheduler.config.prediction_type == "epsilon"
    assert pipe.scheduler.config.rescale_betas_zero_snr == False
    assert pipe.scheduler.config.sample_max_value == 1.0
    assert pipe.scheduler.config.set_alpha_to_one == False
    assert pipe.scheduler.config.skip_prk_steps == True
    assert pipe.scheduler.config.steps_offset == 1
    assert pipe.scheduler.config.timestep_spacing == "leading"
    assert pipe.scheduler.config.trained_betas == None
    assert pipe.scheduler.config.use_karras_sigmas == False

    assert change_scheduler_or_not(pipe, "UniPCMultistepScheduler")

    assert isinstance(pipe.scheduler, UniPCMultistepScheduler)
    assert pipe.scheduler.config._class_name == "UniPCMultistepScheduler"
    assert pipe.scheduler.config._diffusers_version == "0.22.0.dev0"
    assert pipe.scheduler.config.beta_end == 0.012
    assert pipe.scheduler.config.beta_schedule == "scaled_linear"
    assert pipe.scheduler.config.beta_start == 0.00085
    assert pipe.scheduler.config.clip_sample == False
    assert pipe.scheduler.config.interpolation_type == "linear"
    assert pipe.scheduler.config.num_train_timesteps == 1000
    assert pipe.scheduler.config.prediction_type == "epsilon"
    assert pipe.scheduler.config.rescale_betas_zero_snr == False
    assert pipe.scheduler.config.sample_max_value == 1.0
    assert pipe.scheduler.config.set_alpha_to_one == False
    assert pipe.scheduler.config.skip_prk_steps == True
    assert pipe.scheduler.config.steps_offset == 1
    assert pipe.scheduler.config.timestep_spacing == "leading"
    assert pipe.scheduler.config.trained_betas == None
    assert pipe.scheduler.config.use_karras_sigmas == False
