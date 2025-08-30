from pathlib import Path

from cliimage.ApplicationPaths import ApplicationPaths
from cliimage.Core import ProcessConfigurations

import torch

application_path = Path(__file__).resolve().parents[3]

def test_ProcessConfigurations():

    application_paths = ApplicationPaths.create(
        configpath=application_path)

    process_configurations = ProcessConfigurations(application_paths)

    process_configurations.process_configurations()

    assert process_configurations.configurations[
        "batch_processing_configuration"] is not None
    assert process_configurations.configurations[
        "flux_generation_configuration"] is not None
    assert process_configurations.configurations["model_list"] is not None
    assert process_configurations.configurations[
        "nunchaku_configuration"] is not None
    assert process_configurations.configurations[
        "nunchaku_loras_configuration"] is not None
    assert process_configurations.configurations["pipeline_inputs"] is not None

    # User's configuration specific.
    batch_processing_configuration = process_configurations.configurations[
        "batch_processing_configuration"]

    assert batch_processing_configuration.number_of_images == 1

    nunchaku_configuration = process_configurations.configurations[
        "nunchaku_configuration"]

    assert nunchaku_configuration.cuda_device == "cuda:0"
    assert nunchaku_configuration.torch_dtype == torch.bfloat16

    flux_generation_configuration = process_configurations.configurations[
        "flux_generation_configuration"]

    assert flux_generation_configuration.height == 1216
    assert flux_generation_configuration.temporary_save_path == Path(
        "/Data/Private/")

    nunchaku_loras_configuration = process_configurations.configurations[
        "nunchaku_loras_configuration"]

    assert len(nunchaku_loras_configuration.loras) == 1
    lora_name = next(iter(nunchaku_loras_configuration.loras))
    assert nunchaku_loras_configuration.loras[lora_name].lora_strength == 0.25
