from corecode.Utilities import (
    DataSubdirectories,
    )

from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler

from morediffusers.Configurations import (
    Configuration,
    IPAdapterConfiguration)
from morediffusers.Schedulers import create_ddim_scheduler

from morediffusers.Wrappers import (
    load_ip_adapter,
    change_pipe_with_ip_adapter_to_cuda_or_not)

from morediffusers.Wrappers.models import create_motion_adapter
from morediffusers.Wrappers.pipelines import create_animate_diff_sdxl_pipeline
from morediffusers.Wrappers import change_pipe_to_cuda_or_not

from pathlib import Path

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[3] / "TestData"


def test_create_animate_diff_sdxl_pipeline_creates_with_motion_adapter():

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    pretrained_diffusion_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "dataautogpt3" / "ProteusV0.5"

    adapter = create_motion_adapter(
        pretrained_model_name_or_path,
        )

    pipeline = create_animate_diff_sdxl_pipeline(
        pretrained_diffusion_model_name_or_path,
        motion_adapter=adapter
        )

    assert isinstance(pipeline.scheduler, EulerDiscreteScheduler)
    assert isinstance(pipeline.vae, AutoencoderKL)
    assert pipeline.text_encoder != None
    assert pipeline.text_encoder_2 != None

def test_animate_diff_sdxl_pipeline_can_change_scheduler():

    test_file_path = test_data_directory / "configuration_for_video.yml"
    assert test_file_path.exists()

    configuration = Configuration(test_file_path)

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    adapter = create_motion_adapter(pretrained_model_name_or_path)

    pipeline = create_animate_diff_sdxl_pipeline(
        configuration.diffusion_model_path,
        motion_adapter=adapter
        )

    original_scheduler_name = pipeline.scheduler.config._class_name

    assert original_scheduler_name == "EulerAncestralDiscreteScheduler"

    scheduler = create_ddim_scheduler(
        configuration.diffusion_model_path,
        subfolder="scheduler")

    pipeline.scheduler = scheduler

    assert isinstance(pipeline.scheduler, DDIMScheduler)

    changed_scheduler_name = pipeline.scheduler.config._class_name

    assert changed_scheduler_name == "DDIMScheduler"

def test_animate_diff_sdxl_pipeline_can_use_cuda():

    test_file_path = test_data_directory / "configuration_for_video.yml"
    assert test_file_path.exists()

    configuration = Configuration(test_file_path)

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    adapter = create_motion_adapter(pretrained_model_name_or_path)

    pipeline = create_animate_diff_sdxl_pipeline(
        configuration.diffusion_model_path,
        motion_adapter=adapter,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload
        )

    change_pipe_to_cuda_or_not(configuration, pipeline)

    assert True

    scheduler = create_ddim_scheduler(
        configuration.diffusion_model_path,
        subfolder="scheduler")

    pipeline.scheduler = scheduler

    change_pipe_to_cuda_or_not(configuration, pipeline)

    assert True

def test_animate_diff_sdxl_pipeline_can_load_ip_adapter():

    test_file_path = test_data_directory / "configuration_for_video.yml"
    ip_adapter_file_path = test_data_directory / "ip_adapter_configuration_single.yml"

    configuration = Configuration(test_file_path)
    ip_adapter_configuration = IPAdapterConfiguration(ip_adapter_file_path)

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    adapter = create_motion_adapter(pretrained_model_name_or_path)

    pipeline = create_animate_diff_sdxl_pipeline(
        configuration.diffusion_model_path,
        motion_adapter=adapter,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload
        )

    scheduler = create_ddim_scheduler(
        configuration.diffusion_model_path,
        subfolder="scheduler")

    pipeline.scheduler = scheduler

    change_pipe_to_cuda_or_not(configuration, pipeline)

    load_ip_adapter(pipeline, ip_adapter_configuration)

    change_pipe_with_ip_adapter_to_cuda_or_not(
        pipeline,
        ip_adapter_configuration)

    assert True