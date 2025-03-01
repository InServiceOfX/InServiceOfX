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
    StableDiffusionXLUserInput
)

from morediffusers.Configurations import (
    LoRAsConfigurationForMoreDiffusers,
    StableDiffusionXLGenerationConfiguration,
    DiffusionPipelineConfiguration
)

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import create_seed_generator

from morediffusers.Wrappers.pipelines import (
    create_stable_diffusion_xl_pipeline,
    change_pipe_to_cuda_or_not,
    load_loras
)

def terminal_only_finite_loop_sdxl_with_loras():

    configuration = DiffusionPipelineConfiguration()
    generation_configuration = StableDiffusionXLGenerationConfiguration()

    user_input = StableDiffusionXLUserInput(generation_configuration)

    start_time = time.time()

    pipe = create_stable_diffusion_xl_pipeline(configuration)

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

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipe)

    for index in range(user_input.iterations):

        generation_kwargs = user_input.get_generation_kwargs()

        generation_kwargs.update(
            generation_configuration.get_generation_kwargs())

        generation_kwargs["generator"] = create_seed_generator(
            configuration,
            generation_configuration)

        image = pipe(**generation_kwargs).images[0]

        create_image_filename_and_save(
            user_input,
            index,
            image,
            generation_configuration,
            Path(configuration.diffusion_model_path).name)

        user_input.update_guidance_scale()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_finite_loop_sdxl_with_loras()