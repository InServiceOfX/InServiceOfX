"""
@brief Generate N (hence "finite") number of images in a for loop (hence
"loop"). Run this in your terminal, command prompt (hence "terminal only").
"""

from collections import namedtuple
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

from corecode.Utilities import (
    clear_torch_cache_and_collect_garbage,
    get_user_input,
    FloatParameter,
    IntParameter,
    StringParameter)
from morediffusers.Schedulers import change_scheduler_or_not

from morediffusers.Wrappers import (
    create_stable_diffusion_xl_pipeline,
    load_loras)

from morediffusers.Configurations import Configuration
from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers

def format_float_for_string(value):
    if value == int(value):
        return f"{int(value)}"
    else:
        # Truncate to 3 places, remove trailing zeros
        return f"{value:.3f}".rstrip('0').rstrip('.')

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
        configuration.scheduler)

    end_time = time.time()
    duration = end_time - start_time

    changed_scheduler_name = pipe.scheduler.config._class_name

    print("-------------------------------------------------------------------")
    print(f"Completed pipeline creation, took {duration:.2f} seconds.")
    print("-------------------------------------------------------------------")

    if is_scheduler_changed:
        print(
            "\nDiagnostic: scheduler changed, originally: ",
            original_scheduler_name,
            "\nNow: ",
            changed_scheduler_name)
    else:
        print(
            "\nDiagnostic: scheduler didn't change, originally: ",
            original_scheduler_name,
            "\nStayed: ",
            changed_scheduler_name)

    print("\nDiagnostic: pipe.unet.config.time_cond_proj_dim: ")
    print(pipe.unet.config.time_cond_proj_dim)

    #
    #
    # LoRAs - Low Rank Adaptations
    #
    #
    loras_configuration = LoRAsConfigurationForMoreDiffusers()
    load_loras(pipe, loras_configuration)

    print("\n LoRAs: \n")
    print(pipe.get_active_adapters())
    print(pipe.get_list_adapters())

    prompt = StringParameter(get_user_input(str, "Prompt: "))
    prompt_2 = StringParameter(get_user_input(str, "Prompt 2: ", ""))
    if prompt_2 == "":
        prompt_2 = None

    # Example negative prompt:
    # "(lowres, low quality, worst quality:1.2), (text:1.2), glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
    # prompt for what you want to not include.
    negative_prompt = StringParameter(
        get_user_input(str, "Negative prompt: ", ""))

    negative_prompt_2 = StringParameter(
        get_user_input(str, "Negative prompt 2: ", ""))
    if negative_prompt_2 == "":
        negative_prompt_2 = None

    number_of_steps = IntParameter(
        get_user_input(int, "Number of steps, normally 50"))

    print("prompt: ", prompt.value)
    print("negative prompt: ", negative_prompt.value)
    print("Number of Steps: ", number_of_steps.value)

    base_filename = StringParameter(
        get_user_input(
            str,
            "Filename 'base', phrase common in the filenames"))

    iterations = IntParameter(
        get_user_input(int, "Number of Iterations: ", 2))

    model_name = Path(configuration.diffusion_model_path).name

    guidance_scale = configuration.guidance_scale
    guidance_scale_step = 0.0

    # Assume that if guidance scale was indeed set in the configuration, then
    # the user has intention of changing it.
    if guidance_scale is not None:

        guidance_scale_step = FloatParameter(
            get_user_input(
                float,
                "Guidance scale step value, enter small decimal value",
                0.0))


    cross_attention_kwargs=None
    
    if loras_configuration.lora_scale != None:
        cross_attention_kwargs={"scale": float(loras_configuration.lora_scale)}

    for index in range(iterations.value):

        """
        @details See
        diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
        and def __call__(..) for possible arguments.
        """
        if guidance_scale == None:
            # Recall that __call__(..) returns an instance of a
            # StableDiffusionXLPipelineOutput. This is found in
            # diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_output.py
            image = pipe(
                prompt=prompt.value,
                prompt_2=prompt_2.value,
                height=configuration.height,
                width=configuration.width,
                num_inference_steps=number_of_steps.value,
                denoising_end=configuration.denoising_end,
                negative_prompt=negative_prompt.value,
                negative_prompt_2=negative_prompt_2.value,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_skip=configuration.clip_skip
                ).images[0]
        else:
            image = pipe(
                prompt=prompt.value,
                prompt_2=prompt_2.value,
                height=configuration.height,
                width=configuration.width,
                num_inference_steps=number_of_steps.value,
                denoising_end=configuration.denoising_end,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt.value,
                negative_prompt_2=negative_prompt_2.value,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_skip=configuration.clip_skip
                ).images[0]


        filename = ""

        if guidance_scale is None:

            filename = (
                f"{base_filename.value}{model_name}-"
                f"Steps{number_of_steps.value}Iter{index}"
            )
        else:

            filename = (
                f"{base_filename.value}{model_name}-"
                f"Steps{number_of_steps.value}Iter{index}Guidance{format_float_for_string(guidance_scale)}"
            )

        image_format = image.format if image.format else "PNG"
        file_path = Path(configuration.temporary_save_path) / \
            f"{filename}.{image_format.lower()}"
        image.save(file_path)
        print(f"Image saved to {file_path}")

        # Update parameters for iterative steps.
        if guidance_scale is not None:
            guidance_scale += guidance_scale_step.value

    clear_torch_cache_and_collect_garbage()


if __name__ == "__main__":

    terminal_only_finite_loop_main_with_loras()