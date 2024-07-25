from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Wrappers import create_stable_diffusion_xl_pipeline

data_sub_dirs = DataSubdirectories()

def test_create_stable_diffusion_xl_pipeline_no_cpu_offload():

    pipe = create_stable_diffusion_xl_pipeline(
        data_sub_dirs.ModelsDiffusion / "stabilityai" / "stable-diffusion-xl-base-1.0",
        None,
        is_enable_cpu_offload=False,
        is_enable_sequential_cpu_offload=False
        )

    assert pipe.scheduler.config._class_name == "EulerDiscreteScheduler"
    assert pipe.unet.config.time_cond_proj_dim == None