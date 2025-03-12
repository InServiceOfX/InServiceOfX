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
    )

from morediffusers.Configurations import (
    DiffusionPipelineConfiguration,
    FluxGenerationConfiguration,
    NunchakuLoRAsConfigurationForMoreDiffusers)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    transformer_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

def terminal_only_finite_loop_flux_and_nunchaku():

    configuration = DiffusionPipelineConfiguration(
        DiffusionPipelineConfiguration.DEFAULT_CONFIG_PATH.parent / \
            "flux_pipeline_configuration.yml")

    generation_configuration = FluxGenerationConfiguration()

    user_input = FluxPipelineUserInput(generation_configuration)

    if generation_configuration.seed is None:
        print("generation_configuration.seed is None")
    else:
        print("generation_configuration.seed: ", generation_configuration.seed)

    generation_configuration.max_sequence_length = 512

    start_time = time.time()

    path = Path(configuration.diffusion_model_path).parents[1] / \
        "mit-han-lab" / "svdq-flux.1-t5"

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        path,
        configuration)

    pipeline = text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
        configuration.diffusion_model_path,
        configuration,
        text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    prompt_embeds, pooled_prompt_embeds, text_ids = \
        text_encoder_2_inference.encode_prompt(
            pipeline,
            generation_configuration,
            user_input.prompt,
            user_input.prompt_2)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = Path(configuration.diffusion_model_path).parents[1] / \
        "mit-han-lab" / "svdq-int4-flux.1-dev"

    transformer = transformer_inference.create_flux_transformer(path)

    pipeline = transformer_inference.create_flux_transformer_pipeline(
        configuration.diffusion_model_path,
        configuration,
        transformer)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    loras_configuration = NunchakuLoRAsConfigurationForMoreDiffusers()

    valid_loras = loras_configuration.get_valid_loras()

    for lora in valid_loras:
        print(str(lora[0]))
        transformer.update_lora_params(str(lora[0]))
        transformer.set_lora_strength(lora[1])

    if transformer.unquantized_loras is None or \
        transformer.unquantized_loras == {}:
        print(
            "transformer.unquantized_loras: ",
            transformer.unquantized_loras)
    else:
        for element in transformer.unquantized_loras:
            print(element)

    print(
        "transformer.unquantized_state_dict: ",
        transformer.unquantized_state_dict)

    end_time = time.time()

    # Loading directly doesn't work sometimes for some cases because of
    #from morediffusers.Configurations import LoRAsConfigurationForMoreDiffusers
    #from morediffusers.Wrappers.pipelines import load_loras
    # unexpected keys in state dict.
    # loras_configuration = LoRAsConfigurationForMoreDiffusers()
    # resulting_loras_state_dict = load_loras(pipeline, loras_configuration)
    # print("resulting_loras_state_dict: ", resulting_loras_state_dict)

    for index in range(user_input.iterations):

        images = transformer_inference.call_pipeline(
            pipeline,
            prompt_embeds,
            pooled_prompt_embeds,
            configuration,
            generation_configuration).images

        print("len(images): ", len(images))

        create_image_filename_and_save(
            user_input,
            index,
            images[0],
            generation_configuration,
            Path(configuration.diffusion_model_path).name)

        user_input.update_guidance_scale()
        generation_configuration.guidance_scale = \
            user_input.guidance_scale

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_finite_loop_flux_and_nunchaku()