from diffusers.utils import load_image
from pathlib import Path

import sys
import time

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_computer_vision_directory = python_libraries_path / "MoreComputerVision"
more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_computer_vision_directory) in sys.path:
    sys.path.append(str(more_computer_vision_directory))

if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))

from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    DataSubdirectories,
    get_user_input,
    StringParameter
    )

from morecomputervision.Conversions import export_to_mjpg_video

from morediffusers.Applications import (
    print_loras_diagnostics,
    print_pipeline_diagnostics,
)

from morediffusers.Configurations import (
    Configuration,
    GenerateVideoConfiguration,
    IPAdapterConfiguration,
    LoRAsConfigurationForMoreDiffusers)

from morediffusers.Schedulers import create_ddim_scheduler

from morediffusers.Wrappers import (
    change_pipe_to_cuda_or_not,
    change_pipe_with_ip_adapter_to_cuda_or_not,
    change_pipe_with_loras_to_cuda_or_not,
    create_seed_generator,
    load_loras,
    load_ip_adapter)

from morediffusers.Wrappers.models import create_motion_adapter
from morediffusers.Wrappers.pipelines import create_animate_diff_sdxl_pipeline


def terminal_only_animate_diff_sdxl_with_image_prompt():

    data_sub_dirs = DataSubdirectories()

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    adapter = create_motion_adapter(
        pretrained_model_name_or_path,
        )

    start_time = time.time()

    configuration = Configuration()

    pipeline = create_animate_diff_sdxl_pipeline(
        configuration.diffusion_model_path,
        motion_adapter=adapter,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload)

    scheduler = create_ddim_scheduler(
        configuration.diffusion_model_path,
        subfolder="scheduler")

    original_scheduler_name = pipeline.scheduler.config._class_name

    pipeline.scheduler = scheduler

    changed_scheduler_name = pipeline.scheduler.config._class_name

    change_pipe_to_cuda_or_not(configuration, pipeline)

    end_time = time.time()

    print_pipeline_diagnostics(
        end_time - start_time,
        pipeline,
        changed_scheduler_name != original_scheduler_name,
        original_scheduler_name)

    #
    #
    # LoRAs - Low Rank Adaptations
    #
    #
    start_time = time.time()

    loras_configuration = LoRAsConfigurationForMoreDiffusers()
    load_loras(pipeline, loras_configuration)

    change_pipe_with_loras_to_cuda_or_not(pipeline, loras_configuration)

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipeline)

    #
    #
    # IP Adapter
    #
    #
    start_time = time.time()

    ip_adapter_configuration = IPAdapterConfiguration()
    load_ip_adapter(pipeline, ip_adapter_configuration)

    change_pipe_with_ip_adapter_to_cuda_or_not(
        pipeline,
        ip_adapter_configuration)

    end_time = time.time()
    duration = end_time - start_time

    print("-------------------------------------------------------------------")
    print(f"Completed loading IP (Image Prompt) Adapter, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    prompt = StringParameter(get_user_input(str, "Prompt: "))
    prompt_2 = StringParameter(get_user_input(str, "Prompt 2: ", ""))
    if prompt_2 == "":
        prompt_2 = None

    negative_prompt = StringParameter(
        get_user_input(str, "Negative prompt: ", ""))

    negative_prompt_2 = StringParameter(
        get_user_input(str, "Negative prompt 2: ", ""))
    if negative_prompt_2 == "":
        negative_prompt_2 = None

    image_prompt = load_image(ip_adapter_configuration.image_filepath)

    frames_output = None

    generate_video_configuration = GenerateVideoConfiguration()

    generator = None

    if generate_video_configuration.seed != None:

        generator = create_seed_generator(
            configuration,
            generate_video_configuration.seed)

    start_time = time.time()

    # pipeline_animatediff_sdxl.py def __call__(..) expects num_frames: int = 16
    # otherwise, obtain TypeError for 
    # File "/ThirdParty/diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py", line 1101, in __call__
    # latents = self.prepare_latents(
    # File "/ThirdParty/diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py", line 730, in prepare_latents
    # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    # File "/ThirdParty/diffusers/src/diffusers/utils/torch_utils.py", line 81, in randn_tensor
    # latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    # TypeError: randn(): argument 'size' failed to unpack the object at pos 3 with error "type must be tuple of ints,but got NoneType"    
    if generate_video_configuration.num_frames == None:
        generate_video_configuration.num_frames = 16

    if generate_video_configuration.guidance_scale == None:

        frames_output = pipeline(
            prompt=prompt.value,
            prompt_2=prompt_2.value,
            num_frames=generate_video_configuration.num_frames,
            height=configuration.height,
            width=configuration.width,
            num_inference_steps=generate_video_configuration.num_inference_steps,
            denoising_end=configuration.denoising_end,
            negative_prompt=negative_prompt.value,
            negative_prompt_2=negative_prompt_2.value,
            num_videos_per_prompt=generate_video_configuration.num_videos_per_prompt,
            generator=generator,
            ip_adapter_image=image_prompt,
            # TODO: Investigate deprecation
            #cross_attention_kwargs=user_input.cross_attention_kwargs,
            clip_skip=generate_video_configuration.clip_skip
            ).frames
    else:
        try:
            frames_output = pipeline(
                prompt=prompt.value,
                prompt_2=prompt_2.value,
                num_frames=generate_video_configuration.num_frames,
                height=configuration.height,
                width=configuration.width,
                num_inference_steps=generate_video_configuration.num_inference_steps,
                denoising_end=configuration.denoising_end,
                guidance_scale=generate_video_configuration.guidance_scale,
                negative_prompt=negative_prompt.value,
                negative_prompt_2=negative_prompt_2.value,
                num_videos_per_prompt=generate_video_configuration.num_videos_per_prompt,
                generator=generator,
                ip_adapter_image=image_prompt,
                # TODO: Investigate deprecation
                #cross_attention_kwargs=user_input.cross_attention_kwargs,
                clip_skip=generate_video_configuration.clip_skip
                ).frames
        except TypeError as err:
            print("TypeError: ", err)
            return 1

    frames = frames_list[0]

    end_time = time.time()

    print("-------------------------------------------------------------------")
    print(f"Completed frames generation, took {end_time - start_time:.2f} seconds.")
    print("-------------------------------------------------------------------")

    file_path = create_video_file_path(
        configuration,
        generate_video_configuration)

    print("File path for file into export_to_video: ", str(file_path))

    try:
        output_video_path = export_to_mjpg_video(
            frames,
            str(file_path),
            fps=generate_video_configuration.fps)
        print("Exported!\n")
        print("Output video path from export_to_video: ", output_video_path)

    except RuntimeError as err:

        print("RuntimeError: ", err)

    except AttributeError as err:
        print("AttributeError: ", err)

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_animate_diff_sdxl_with_image_prompt()
