"""
@brief Generate N (hence "finite") number of images in a for loop (hence
"loop"). Run this in your terminal, command prompt (hence "terminal only").
"""

from pathlib import Path
import sys
import torch
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
    FluxPipelineUserInput
    )

from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    FluxGenerationConfiguration)
from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers

from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import create_seed_generator

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
    load_loras,
    create_flux_pipeline)

def terminal_only_finite_loop_flux_with_loras():
    # Load configuration
    pipeline_config = DiffusionPipelineConfiguration(
        DiffusionPipelineConfiguration.DEFAULT_CONFIG_PATH.parent / 
        "flux_pipeline_configuration.yml")
    
    generation_config = FluxGenerationConfiguration()
    
    user_input = FluxPipelineUserInput(generation_config)

    start_time = time.time()

    # Create pipeline with LoRAs
    pipe = create_flux_pipeline(pipeline_config)

    original_scheduler_name = pipe.scheduler.config._class_name

    is_scheduler_changed = None
    if (pipeline_config.scheduler != "" and pipeline_config.scheduler != None):
        is_scheduler_changed = change_scheduler_or_not(
            pipe,
            pipeline_config.scheduler,
            pipeline_config.a1111_kdiffusion)

    change_pipe_to_cuda_or_not(pipeline_config, pipe)

    end_time = time.time()

    print_pipeline_diagnostics(
        end_time - start_time,
        pipe,
        is_scheduler_changed,
        original_scheduler_name)

    # Load LoRAs configuration
    start_time = time.time()

    loras_config = LoRAsConfigurationForMoreDiffusers()
    load_loras(pipe, loras_config)

    end_time = time.time()

    print_loras_diagnostics(end_time - start_time, pipe)

    for index in range(user_input.iterations):
        generation_kwargs = user_input.get_generation_kwargs()
        
        generation_kwargs.update(
            generation_config.get_generation_kwargs())

        generation_kwargs["generator"] = create_seed_generator(
            pipeline_config,
            generation_config)

        image = pipe(**generation_kwargs).images[0]

        create_image_filename_and_save(
            user_input,
            index,
            image,
            generation_config,
            Path(pipeline_config.diffusion_model_path).name)

        user_input.update_guidance_scale()

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":
    terminal_only_finite_loop_flux_with_loras()