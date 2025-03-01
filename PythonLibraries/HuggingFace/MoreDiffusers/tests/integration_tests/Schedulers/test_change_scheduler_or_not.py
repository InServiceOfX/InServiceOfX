# TODO: Run these integration tests in sequential order.

from corecode.Utilities import (
    DataSubdirectories,
    )
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    )
from morediffusers.Schedulers import change_scheduler_or_not

import torch

data_sub_dirs = DataSubdirectories()

def test_change_scheduler_or_not_changes_scheduler():
    """
    @ref https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet#text-to-image
    See "Text-to-image" in the documentation for "controlnet" of Diffusers
    """
    controlnet = ControlNetModel.from_pretrained(
        data_sub_dirs.ModelsDiffusion / "lllyasviel" / "sd-controlnet-canny",
        torch_dtype=torch.float16,
        use_local_files=True,
        use_safetensors=True)


    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        data_sub_dirs.ModelsDiffusion / "runwayml" / "stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_local_files=True,
        use_safetensors=True)

    print(pipe.scheduler)

    assert isinstance(pipe.scheduler, PNDMScheduler)
    assert pipe.scheduler.config._class_name == "PNDMScheduler"
    assert pipe.scheduler.config._diffusers_version == "0.6.0"
    assert pipe.scheduler.config.beta_end == 0.012
    assert pipe.scheduler.config.beta_schedule == "scaled_linear"
    assert pipe.scheduler.config.beta_start == 0.00085
    assert pipe.scheduler.config.clip_sample == False
    assert pipe.scheduler.config.num_train_timesteps == 1000
    assert pipe.scheduler.config.prediction_type == "epsilon"
    assert pipe.scheduler.config.set_alpha_to_one == False
    assert pipe.scheduler.config.skip_prk_steps == True
    assert pipe.scheduler.config.steps_offset == 1
    assert pipe.scheduler.config.timestep_spacing == "leading"
    assert pipe.scheduler.config.trained_betas == None

    assert change_scheduler_or_not(pipe, "UniPCMultistepScheduler")

    assert isinstance(pipe.scheduler, UniPCMultistepScheduler)
    assert pipe.scheduler.config._class_name == "UniPCMultistepScheduler"
    assert pipe.scheduler.config._diffusers_version == "0.28.0.dev0"
    assert pipe.scheduler.config.beta_end == 0.012
    assert pipe.scheduler.config.beta_schedule == "scaled_linear"
    assert pipe.scheduler.config.beta_start == 0.00085
    assert pipe.scheduler.config.clip_sample == False
    assert pipe.scheduler.config.num_train_timesteps == 1000
    assert pipe.scheduler.config.prediction_type == "epsilon"
    assert pipe.scheduler.config.set_alpha_to_one == False
    assert pipe.scheduler.config.skip_prk_steps == True
    assert pipe.scheduler.config.steps_offset == 1
    assert pipe.scheduler.config.timestep_spacing == "linspace"
    assert pipe.scheduler.config.trained_betas == None

    # Here's the "or not" part where we have scheduler_name=None.
    assert not change_scheduler_or_not(pipe, None)


def test_change_scheduler_or_not_with_fluently():
    path = data_sub_dirs.ModelsDiffusion / "fluently" / "Fluently-XL-v4"
    assert path.exists()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        path,
        local_files_only=True,
        use_safetensors=True)

    assert isinstance(pipe.scheduler, EulerAncestralDiscreteScheduler)
    assert pipe.scheduler.use_karras_sigmas == False

    change_scheduler_or_not(pipe, "DPMSolverSinglestepScheduler")
    assert isinstance(pipe.scheduler, DPMSolverSinglestepScheduler)
    assert pipe.scheduler.use_karras_sigmas == False
    assert pipe.scheduler.config.algorithm_type == "dpmsolver++"

    change_scheduler_or_not(pipe, "DPMSolverMultistepScheduler", "DPM++ 2M SDE")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert pipe.scheduler.use_karras_sigmas == False
    assert pipe.scheduler.config.algorithm_type == "sde-dpmsolver++"

    change_scheduler_or_not(
        pipe,
        "DPMSolverSinglestepScheduler",
        "DPM++ SDE Karras")
    assert isinstance(pipe.scheduler, DPMSolverSinglestepScheduler)
    assert pipe.scheduler.use_karras_sigmas == True