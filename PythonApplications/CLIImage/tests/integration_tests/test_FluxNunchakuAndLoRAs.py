from pathlib import Path

from cliimage.ApplicationPaths import ApplicationPaths
from cliimage.Core import ProcessConfigurations
from cliimage.Terminal import TerminalUI

from morediffusers.Applications import FluxNunchakuAndLoRAs

application_path = Path(__file__).resolve().parents[2]

def test_FluxNunchakuAndLoRAs_call_pipeline():

    application_paths = ApplicationPaths.create(
        configpath=application_path)

    terminal_ui = TerminalUI()

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)

    process_configurations.process_configurations()

    flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
        process_configurations.configurations["nunchaku_configuration"],
        process_configurations.configurations["flux_generation_configuration"],
        process_configurations.configurations["pipeline_inputs"],
        process_configurations.configurations["nunchaku_loras_configuration"])

    print("Creating prompt embeds...")

    prompt_embeds, pooled_prompt_embeds, text_ids, negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = \
        flux_nunchaku_and_loras.create_prompt_embeds()

    print("Created prompt embeds...")

    print("Deleting text encoder 2 and pipeline...")

    flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()

    print("Deleted text encoder 2 and pipeline...")

    print("Creating transformer and pipeline...")

    flux_nunchaku_and_loras.create_transformer_and_pipeline()

    print("Created transformer and pipeline...")

    images = flux_nunchaku_and_loras.call_pipeline(
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds)

    print("Called pipeline...")

    batch_processing_configuration = \
        process_configurations.configurations["batch_processing_configuration"]

    batch_processing_configuration.create_and_save_image(
        0,
        images[0],
        flux_nunchaku_and_loras._generation_configuration,
        process_configurations.get_model_name())

    print("Created and saved image...")

    assert images is not None

def test_FluxNunchakuAndLoRAs_batch_processing():

    application_paths = ApplicationPaths.create(
        configpath=application_path)

    terminal_ui = TerminalUI()

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)

    process_configurations.process_configurations()

    flux_nunchaku_and_loras = FluxNunchakuAndLoRAs(
        process_configurations.configurations["nunchaku_configuration"],
        process_configurations.configurations["flux_generation_configuration"],
        process_configurations.configurations["pipeline_inputs"],
        process_configurations.configurations["nunchaku_loras_configuration"])

    flux_nunchaku_and_loras.create_prompt_embeds()

    flux_nunchaku_and_loras.delete_text_encoder_2_and_pipeline()

    flux_nunchaku_and_loras.create_transformer_and_pipeline()

    batch_processing_configuration = \
        process_configurations.get_batch_processing_configuration()

    print(
        "Batch processing, number of images: ",
        batch_processing_configuration.number_of_images)

    for index in range(batch_processing_configuration.number_of_images):
        images = flux_nunchaku_and_loras.call_pipeline_with_prompt_embed(0)

        if images is not None:
            batch_processing_configuration.create_and_save_image(
                index,
                images[0],
                flux_nunchaku_and_loras._generation_configuration,
                process_configurations.get_model_name())
        else:
            print("Pipeline execution failed!")
            print("Index: ", index)

        flux_nunchaku_and_loras._generation_configuration.guidance_scale += \
            batch_processing_configuration.guidance_scale_step

    print("Batch processing complete...")

    assert True