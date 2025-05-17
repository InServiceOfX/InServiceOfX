# TODO: Run these integration tests in sequential order.

from corecode.Utilities import (
    DataSubdirectories,
    )
from morediffusers.Schedulers import CreateSchedulerMap
from moreinstantid.Configuration import Configuration
from moreinstantid.Wrappers import (
    create_controlnet,
    create_stable_diffusion_xl_pipeline
    )
from pathlib import Path
from diffusers import EulerAncestralDiscreteScheduler

import pytest
import torch

test_data_directory = Path(__file__).resolve().parent.parent.parent / "TestData"

def test_create_stable_diffusion_xl_pipeline_constructs():
    test_file_path = test_data_directory / "integration_configuration_0.yml"
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

def test_create_stable_diffusion_xl_pipeline_can_change_scheduler():

    test_file_path = test_data_directory / "integration_configuration_0.yml"
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

    schedulers_map = CreateSchedulerMap.get_map()

    pipe.scheduler = schedulers_map["UniPCMultistepScheduler"].from_config(
        pipe.scheduler.config)

    assert isinstance(pipe.scheduler, schedulers_map["UniPCMultistepScheduler"])
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

def test_created_pipeline_can_load_lora():

    test_file_path = test_data_directory / "integration_configuration_0.yml"
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

    load_scale = 0.85

    data_subdirectories = DataSubdirectories()

    pipe.load_lora_weights(
        data_subdirectories.ModelsDiffusion / "CiroN2022" / "toy-face",
        weight_name="toy_face_sdxl.safetensors",
        adapter_name="toy"
        )

    active_adapters = pipe.get_active_adapters()
    assert isinstance(active_adapters, list)
    assert len(active_adapters) == 1
    assert active_adapters[0] == "toy"

    list_adapters_component_wise = pipe.get_list_adapters()
    assert isinstance(list_adapters_component_wise, dict)
    assert len(list_adapters_component_wise) == 1
    assert "unet" in list_adapters_component_wise.keys()
    assert list_adapters_component_wise["unet"] == ['toy']

def test_created_pipeline_can_set_another_lora_adapter():

    test_file_path = test_data_directory / "integration_configuration_0.yml"
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

    load_scale = 0.85

    data_subdirectories = DataSubdirectories()

    pipe.load_lora_weights(
        data_subdirectories.ModelsDiffusion / "CiroN2022" / "toy-face",
        weight_name="toy_face_sdxl.safetensors",
        adapter_name="toy"
        )

    pipe.load_lora_weights(
        data_subdirectories.ModelsDiffusion / "nerijs" / "pixel-art-xl",
        weight_name="pixel-art-xl.safetensors",
        adapter_name="pixel")

    pipe.set_adapters("pixel")

    active_adapters = pipe.get_active_adapters()
    assert isinstance(active_adapters, list)
    assert len(active_adapters) == 1
    assert active_adapters[0] == "pixel"

    list_adapters_component_wise = pipe.get_list_adapters()
    assert isinstance(list_adapters_component_wise, dict)
    assert len(list_adapters_component_wise) == 1
    assert "unet" in list_adapters_component_wise.keys()
    assert list_adapters_component_wise["unet"] == ["toy", "pixel"]
