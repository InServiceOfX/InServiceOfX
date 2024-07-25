from pathlib import Path

import sys
import time

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
morecomputervision_directory = python_libraries_path / "MoreComputerVision"

more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"

more_insightface_directory = \
    python_libraries_path / "ThirdParties" / "MoreInsightFace"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(morecomputervision_directory) in sys.path:
    sys.path.append(str(morecomputervision_directory))

if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))

if not str(more_insightface_directory) in sys.path:
    sys.path.append(str(more_insightface_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from morediffusers.Applications import (
    create_image_filename_and_save,
    print_loras_diagnostics,
    print_pipeline_diagnostics,
    UserInputWithLoras
)

from morediffusers.Configurations import Configuration
from morediffusers.Configurations import IPAdapterConfiguration
from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import (
    change_pipe_to_cuda_or_not,
    change_pipe_with_ip_adapter_to_cuda_or_not,
    change_pipe_with_loras_to_cuda_or_not,
    create_stable_diffusion_xl_pipeline,
    load_loras,
    load_ip_adapter)

from moreinsightface.Configurations import Configuration as MoreInsightFaceConfiguration

from moreinsightface.Wrappers import FaceAnalysisWrapper

def terminal_only_face_embedding():

    start_time = time.time()

    configuration = Configuration()

    pipe = create_stable_diffusion_xl_pipeline(
        configuration.diffusion_model_path,
        configuration.single_file_diffusion_checkpoint,
        configuration.torch_dtype,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload)

    original_scheduler_name = pipe.scheduler.config._class_name

    is_scheduler_changed = change_scheduler_or_not(
        pipe,
        configuration.scheduler,
        configuration.a1111_kdiffusion)

    change_pipe_to_cuda_or_not(configuration, pipe)

    end_time = time.time()

    print_pipeline_diagnostics(
        end_time - start_time,
        pipe,
        is_scheduler_changed,
        original_scheduler_name)

    #
    #
    # LoRAs - Low Rank Adaptations
    #
    #
    start_time = time.time()

    loras_configuration = LoRAsConfigurationForMoreDiffusers()
    load_loras(pipe, loras_configuration)

    change_pipe_with_loras_to_cuda_or_not(pipe, loras_configuration)

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipe)

    #
    #
    # IP Adapter
    #
    #
    start_time = time.time()

    ip_adapter_configuration = IPAdapterConfiguration()
    load_ip_adapter(pipe, ip_adapter_configuration, is_load_image_encoder=False)

    change_pipe_with_ip_adapter_to_cuda_or_not(pipe, ip_adapter_configuration)

    end_time = time.time()
    duration = end_time - start_time

    print("-------------------------------------------------------------------")
    print(f"Completed loading IP (Image Prompt) Adapter, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    #
    #
    # Extract face embeddings with insightface.
    #
    #

    insightface_configuration = MoreInsightFaceConfiguration()

    start_time = time.time()

    face_analysis = FaceAnalysisWrapper(
        name=insightface_configuration.face_analysis_model_name,
        root=insightface_configuration.face_analysis_model_directory_path,
        det_size=insightface_configuration.det_size)

    face_embedding_results = face_analysis.get_face_embedding_from_image(
        ip_adapter_configuration.image_filepath,
        configuration)

    end_time = time.time()
    duration = end_time - start_time

    print("-------------------------------------------------------------------")
    print(f"Completed extracting face embeddings with insightface, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    user_input = UserInputWithLoras(configuration, loras_configuration)

    for index in range(user_input.iterations.value):
        if user_input.guidance_scale == None:

            image = pipe(
                prompt=user_input.prompt.value,
                prompt_2=user_input.prompt_2.value,
                height=configuration.height,
                width=configuration.width,
                num_inference_steps=user_input.number_of_steps.value,
                denoising_end=configuration.denoising_end,
                negative_prompt=user_input.negative_prompt.value,
                negative_prompt_2=user_input.negative_prompt_2.value,
                generator=user_input.generator,
                # See
                # diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
                # for def __call__(..)
                ip_adapter_image_embeds=[face_embedding_results.id_embeds],
                cross_attention_kwargs=user_input.cross_attention_kwargs,
                clip_skip=configuration.clip_skip
                ).images[0]
        else:
            image = pipe(
                prompt=user_input.prompt.value,
                prompt_2=user_input.prompt_2.value,
                height=configuration.height,
                width=configuration.width,
                num_inference_steps=user_input.number_of_steps.value,
                denoising_end=configuration.denoising_end,
                guidance_scale=user_input.guidance_scale,
                generator=user_input.generator,
                # See
                # diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
                # for def __call__(..)
                ip_adapter_image_embeds=[face_embedding_results.id_embeds],
                negative_prompt=user_input.negative_prompt.value,
                negative_prompt_2=user_input.negative_prompt_2.value,
                cross_attention_kwargs=user_input.cross_attention_kwargs,
                clip_skip=configuration.clip_skip
                ).images[0]

        create_image_filename_and_save(user_input, index, image, configuration)

        # Update parameters for iterative steps.
        if user_input.guidance_scale is not None:
            user_input.guidance_scale += user_input.guidance_scale_step.value

    clear_torch_cache_and_collect_garbage()


if __name__ == "__main__":

    terminal_only_face_embedding()