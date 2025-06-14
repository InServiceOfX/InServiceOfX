"""
Suggested environment: run in Docker container for
InServiceOfX/Scripts/DockerBuilds/Builds/Generative/Diffusion

running in directory
/InServiceOfX/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications

python terminal_only_finite_loop_flux_canny_nunchaku_and_loras.py
"""

from pathlib import Path
import sys
from typing import Optional

python_libraries_path = Path(__file__).resolve().parents[4]
corecode_directory = python_libraries_path / "CoreCode"
more_diffusers_directory = \
    python_libraries_path / "HuggingFace" / "MoreDiffusers"

if not str(corecode_directory) in sys.path:
    sys.path.append(str(corecode_directory))

if not str(more_diffusers_directory) in sys.path:
    sys.path.append(str(more_diffusers_directory))

from corecode.Utilities import clear_torch_cache_and_collect_garbage

from diffusers import FluxControlPipeline
from diffusers.utils import load_image

from controlnet_aux import CannyDetector

from morediffusers.Applications import (
    create_image_filename_and_save,
    FluxPipelineUserInput,
    )

from morediffusers.Configurations import (
    CannyDetectorConfiguration,
    FluxGenerationConfiguration,
    LoRAsConfigurationForMoreDiffusers,
    NunchakuFluxControlConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    )

from morediffusers.Wrappers import create_seed_generator

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

from nunchaku import NunchakuFluxTransformer2dModel

def get_valid_filename(generation_configuration) -> Optional[Path]:
    """
    Get a valid filename from user input, ensuring it exists in the temporary
    save path.
    Returns None if user exits with Ctrl-C.
    
    Args:
        generation_configuration: Configuration object containing
        temporary_save_path

    Returns:
        Optional[Path]: Valid file path if found, None if user exits
    """
    while True:
        try:
            filename = input("Enter filename for control: ").strip()
            # Skip empty input
            if not filename:
                continue
                
            full_path = Path(generation_configuration.temporary_save_path) / filename
            
            if full_path.exists():
                return full_path
            else:
                print(f"File not found: {full_path}")
                print("Please try again or press Ctrl-C to exit")
                
        except KeyboardInterrupt:
            print("\nExiting filename input...")
            return None

def terminal_only_finite_loop_flux_canny_nunchaku_and_loras():

    configuration = NunchakuFluxControlConfiguration.from_yaml()

    generation_configuration = FluxGenerationConfiguration()

    valid_filename = get_valid_filename(generation_configuration)
    if valid_filename:
        print(f"Using existing file: {valid_filename}")
    else:
        print("No valid filename provided, exiting...")
        return

    canny_detector_configuration = CannyDetectorConfiguration.from_yaml()

    processor = CannyDetector()

    control_image = load_image(str(valid_filename))

    control_image = processor(
        control_image,
        **canny_detector_configuration.to_kwargs())

    del processor

    clear_torch_cache_and_collect_garbage()

    user_input = FluxPipelineUserInput(generation_configuration)

    if generation_configuration.seed is None:
        print("generation_configuration.seed is None")
    else:
        print("generation_configuration.seed: ", generation_configuration.seed)

    generation_configuration.max_sequence_length = 512

    path = configuration.nunchaku_t5_model_path

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        path,
        configuration)

    pipeline = \
        text_encoder_2_inference.create_flux_control_text_encoder_2_pipeline(
            configuration.flux_model_path,
            configuration,
            text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    loras_config = LoRAsConfigurationForMoreDiffusers()

    prompt_embeds, pooled_prompt_embeds, text_ids = \
        text_encoder_2_inference.flux_control_encode_prompt(
            pipeline,
            configuration,
            generation_configuration,
            user_input.prompt,
            prompt2=user_input.prompt_2,
            lora_scale=loras_config.lora_scale)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = configuration.nunchaku_flux_model_path

    print("path: ", path)
    print("path.exists(): ", path.exists())

    print("run NunchakuFluxTransformer2dModel.from_pretrained()")

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(str(path))

    flux_dev_path = configuration.flux_model_path

    print("flux_dev_path: ", flux_dev_path)

    for _, lora_parameters in loras_config.loras.items():
        path = lora_parameters["directory_path"] + \
            "/" + \
            lora_parameters["weight_name"]
        print("path: ", path)
        print("path.exists(): ", Path(path).exists())
        weight = lora_parameters["adapter_weight"]
        print("weight: ", weight)

        transformer.update_lora_params(str(path))
        transformer.set_lora_strength(float(weight))

    pipe = FluxControlPipeline.from_pretrained(
        str(flux_dev_path),
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=configuration.torch_dtype)

    change_pipe_to_cuda_or_not(configuration, pipe)

    print("temporary_save_path: ", generation_configuration.temporary_save_path)

    for index in range(user_input.iterations):

        image = pipe(
            control_image=control_image,
            height=generation_configuration.height,
            width=generation_configuration.width,
            num_inference_steps=user_input.num_inference_steps,
            guidance_scale=user_input.guidance_scale,
            generator=create_seed_generator(
                configuration,
                generation_configuration),
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            ).images[0]

        create_image_filename_and_save(
            user_input,
            index,
            image,
            generation_configuration,
            Path(configuration.flux_model_path).name)

        user_input.update_guidance_scale()
        generation_configuration.guidance_scale = \
            user_input.guidance_scale

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":
    terminal_only_finite_loop_flux_canny_nunchaku_and_loras()