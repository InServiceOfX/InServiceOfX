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
    FluxGenerationConfiguration,
    NunchakuConfiguration,
    NunchakuLoRAsConfiguration)

from morediffusers.NunchakuWrappers import (
    text_encoder_2_inference,
    transformer_inference,
    )

from morediffusers.Wrappers.pipelines import (
    change_pipe_to_cuda_or_not,
)

from nunchaku import NunchakuFluxTransformer2dModel

def terminal_only_finite_loop_flux_nunchaku_and_loras():

    configuration = NunchakuConfiguration.from_yaml(
        NunchakuConfiguration.get_default_config_path())

    generation_configuration = FluxGenerationConfiguration.from_yaml(
        FluxGenerationConfiguration.get_default_config_path())

    user_input = FluxPipelineUserInput(generation_configuration)

    if generation_configuration.seed is None:
        print("generation_configuration.seed is None")
        generation_configuration.seed = 0
    else:
        print("generation_configuration.seed: ", generation_configuration.seed)

    generation_configuration.max_sequence_length = 512

    path = configuration.nunchaku_t5_model_path

    start_time = time.time()

    text_encoder_2 = text_encoder_2_inference.create_flux_text_encoder_2(
        path,
        configuration)

    pipeline = text_encoder_2_inference.create_flux_text_encoder_2_pipeline(
        configuration.flux_model_path,
        configuration,
        text_encoder_2)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    loras_configuration = NunchakuLoRAsConfiguration.from_yaml(
        NunchakuLoRAsConfiguration.get_default_config_path())

    prompt_embeds, pooled_prompt_embeds, text_ids = \
        text_encoder_2_inference.encode_prompt(
            pipeline=pipeline,
            generation_configuration=generation_configuration,
            prompt=user_input.prompt,
            prompt2=user_input.prompt_2,
            device=configuration.cuda_device,
            lora_scale=loras_configuration.lora_scale)

    negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
        text_encoder_2_inference.encode_prompt(
            pipeline=pipeline,
            generation_configuration=generation_configuration,
            prompt=user_input.negative_prompt,
            prompt2=user_input.negative_prompt_2,
            device=configuration.cuda_device,
            lora_scale=loras_configuration.lora_scale)

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del text_encoder_2

    clear_torch_cache_and_collect_garbage()

    path = configuration.nunchaku_model_paths[0]

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(str(path))

    pipeline = transformer_inference.create_flux_transformer_pipeline(
        str(configuration.flux_model_path),
        configuration,
        transformer)

    change_pipe_to_cuda_or_not(configuration, pipeline)

    valid_loras = loras_configuration.get_valid_loras()

    for lora in valid_loras:
        print(str(lora[0]))
        transformer.update_lora_params(str(lora[0]))
        transformer.set_lora_strength(lora[1])

    if transformer._unquantized_part_loras is None or \
        transformer._unquantized_part_loras == {}:
        print(
            "transformer._unquantized_part_loras: ",
            transformer._unquantized_part_loras)
    else:
        print(
            "transformer._unquantized_part_loras: ",
            len(transformer._unquantized_part_loras))
        for element in transformer._unquantized_part_loras.keys():
            print(element)
    # TODO: This may have been removed by 2.0.0; it was in 1.4.1.
    # print(
    #     "transformer.unquantized_state_dict: ",
    #     transformer.unquantized_state_dict)

    print("Time for encoding prompt: ", time.time() - start_time)

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
            generation_configuration,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds).images

        print("len(images): ", len(images))

        create_image_filename_and_save(
            user_input,
            index,
            images[0],
            generation_configuration,
            Path(configuration.nunchaku_model_paths[0]).name)

        user_input.update_guidance_scale()
        generation_configuration.guidance_scale = \
            user_input.guidance_scale

    clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

    terminal_only_finite_loop_flux_nunchaku_and_loras()