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
    create_image_filenames_and_save_images,
    print_pipeline_diagnostics,
    FluxPipelineUserInput
    )

from morediffusers.Configurations import FluxPipelineConfiguration

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
    create_flux_pipeline
    )


def terminal_only_finite_loop_flux():

    start_time = time.time()

    configuration = FluxPipelineConfiguration()

    pipe = create_flux_pipeline(
        configuration.diffusion_model_path,
        torch_dtype=configuration.torch_dtype,
        variant=configuration.variant,
        use_safetensors=configuration.variant,
        is_enable_cpu_offload=configuration.is_enable_cpu_offload,
        is_enable_sequential_cpu_offload=configuration.is_enable_sequential_cpu_offload)

    original_scheduler_name = pipe.scheduler.config._class_name

    is_scheduler_changed = None
    if (configuration.scheduler != "" and configuration.scheduler != None):
        is_scheduler_changed = change_scheduler_or_not(
            pipe,
            configuration.scheduler,
            configuration.a1111_kdiffusion)

    change_pipe_to_cuda_or_not(configuration, pipe)

    end_time = time.time()

    user_input = FluxPipelineUserInput(configuration)

    for index in range(user_input.iterations.value):

        """
        @details See
        diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        and def __call__(..) for possible arguments.
        """

        images = pipe(
            prompt=user_input.prompt.value,
            prompt_2=user_input.prompt_2.value,
            height=configuration.height,
            width=configuration.width,
            num_inference_steps=user_input.number_of_steps.value,
            guidance_scale=user_input.guidance_scale,
            generator=user_input.generator,
            max_sequence_length=configuration.max_sequence_length,
            ).images

        number_of_images = len(images)

        if number_of_images == 1:

            create_image_filename_and_save(
                user_input,
                index,
                images[0],
                configuration)

        else:

            for i in range(number_of_images):

                create_image_filenames_and_save_images(
                    user_input,
                    index,
                    images,
                    configuration)

        # Update parameters for iterative steps.
        if user_input.guidance_scale is not None:
            user_input.guidance_scale += user_input.guidance_scale_step.value

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_finite_loop_flux()