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
    FluxPipelineUserInput,
    print_pipeline_diagnostics
    )

from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    FluxGenerationConfiguration
)

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import create_seed_generator

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
    create_flux_pipeline
)

def terminal_only_finite_loop_flux():

    configuration = DiffusionPipelineConfiguration(
        DiffusionPipelineConfiguration.DEFAULT_CONFIG_PATH.parent / \
            "flux_pipeline_configuration.yml")

    generation_configuration = FluxGenerationConfiguration()

    user_input = FluxPipelineUserInput(generation_configuration)

    start_time = time.time()

    pipe = create_flux_pipeline(configuration)

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


    for index in range(user_input.iterations):

        """
        @details See
        diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        and def __call__(..) for possible arguments.

        max_sequence_length: Maximum sequence length to use with the 'prompt',
        empirically it was found that 512 is the maximum that can be used
        without a runtime error.
        """

        generation_kwargs = user_input.get_generation_kwargs()
        generation_kwargs.update(
            generation_configuration.get_generation_kwargs())

        generation_kwargs["generator"] = create_seed_generator(
            configuration,
            generation_configuration)

        # From pipeline_flux.py of diffusers, __call__(..) function,
        # Guidance Scale defined in
        # [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
        # guidance_scale defined as 'w' of equation 2. of
        # [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Higher
        # guidance scale encourages to generate images closely linked to
        # text `prompt`, usually at expense of lower image quality.
        images = pipe(**generation_kwargs).images

        print("len(images): ", len(images))

        create_image_filename_and_save(
            user_input,
            index,
            images[0],
            generation_configuration,
            Path(configuration.diffusion_model_path).name)

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_finite_loop_flux()