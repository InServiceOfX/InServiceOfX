# TODO: Run these integration tests in sequential order.

from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Wrappers import load_loras
from morediffusers.Configurations import LoRAsConfiguration
from moreinstantid.Configuration import Configuration
from moreinstantid.Wrappers import (
    create_controlnet,
    create_stable_diffusion_xl_pipeline,
    )
from pathlib import Path
from diffusers import EulerAncestralDiscreteScheduler

import pytest
import torch

test_data_directory = Path(__file__).resolve().parent.parent.parent / "TestData"

def test_load_lora_loads_single_lora():

    test_file_path = test_data_directory / "integration_configuration_0.yml"
    assert test_file_path.exists()

    test_loras_path = test_data_directory / "integration_loras_configuration_1.yml"
    assert test_loras_path.exists()

    configuration = Configuration(test_file_path)
    loras_configuration = LoRAsConfiguration(test_loras_path)

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

    load_loras(pipe, loras_configuration)

    active_adapters = pipe.get_active_adapters()

    assert isinstance(active_adapters, list)
    assert len(active_adapters) == 1
    assert active_adapters[0] == "toy"

    list_adapters_component_wise = pipe.get_list_adapters()
    assert isinstance(list_adapters_component_wise, dict)
    assert len(list_adapters_component_wise) == 1
    assert list_adapters_component_wise["unet"] == ["toy"]


def test_load_lora_loads_two_loras():

    test_file_path = test_data_directory / "integration_configuration_0.yml"

    test_loras_path = test_data_directory / "integration_loras_configuration_0.yml"
    assert test_loras_path.exists()

    configuration = Configuration(test_file_path)
    loras_configuration = LoRAsConfiguration(test_loras_path)

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

    load_loras(pipe, loras_configuration)

    active_adapters = pipe.get_active_adapters()

    assert isinstance(active_adapters, list)
    assert len(active_adapters) == 2
    assert "toy" in active_adapters
    assert "pixel_art" in active_adapters

    list_adapters_component_wise = pipe.get_list_adapters()
    assert isinstance(list_adapters_component_wise, dict)
    assert len(list_adapters_component_wise) == 1
    assert "unet" in list_adapters_component_wise.keys()
    assert "toy" in list_adapters_component_wise["unet"]
    assert "pixel_art" in list_adapters_component_wise["unet"]
