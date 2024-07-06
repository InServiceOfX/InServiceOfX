"""
@brief Generate N (hence "finite") number of images in a for loop (hence
"loop"). Run this in your terminal, command prompt (hence "terminal only").
"""

from pathlib import Path
import sys
import time

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from morediffusers.Applications import (
    create_image_filename_and_save,
    print_loras_diagnostics,
    print_pipeline_diagnostics,
    UserInputWithLoras
    )

from morediffusers.Configurations import Configuration
from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import (
    create_stable_diffusion_xl_pipeline,
    load_loras)


def terminal_only_finite_loop_main_with_loras():

    start_time = time.time()

    configuration = Configuration()

    pipe = create_stable_diffusion_xl_pipeline(
        configuration.diffusion_model_path,
        configuration.single_file_diffusion_checkpoint,
        is_enable_cpu_offload=True,
        is_enable_sequential_cpu=True)

    original_scheduler_name = pipe.scheduler.config._class_name

    is_scheduler_changed = change_scheduler_or_not(
        pipe,
        configuration.scheduler,
        configuration.a1111_kdiffusion)

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

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipe)

    user_input = UserInputWithLoras(configuration, loras_configuration)

    for index in range(user_input.iterations.value):

        """
        @details See
        diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        and def __call__(..) for possible arguments.
        """

        if user_input.guidance_scale == None:
            # Recall that __call__(..) returns an instance of a
            # StableDiffusionXLPipelineOutput. This is found in
            # diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_output.py
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
                negative_prompt=user_input.negative_prompt.value,
                negative_prompt_2=user_input.negative_prompt_2.value,
                cross_attention_kwargs=user_input.cross_attention_kwargs,
                clip_skip=configuration.clip_skip
                ).images[0]

        create_image_filename_and_save(
            user_input,
            index,
            image,
            configuration)

        # Update parameters for iterative steps.
        if user_input.guidance_scale is not None:
            user_input.guidance_scale += user_input.guidance_scale_step.value

    clear_torch_cache_and_collect_garbage()


if __name__ == "__main__":

    terminal_only_finite_loop_main_with_loras()