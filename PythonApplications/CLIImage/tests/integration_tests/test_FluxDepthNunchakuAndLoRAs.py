from cliimage.ApplicationPaths import ApplicationPaths
from cliimage.Core import ProcessConfigurations
from cliimage.Terminal import TerminalUI

from morediffusers.Applications import FluxDepthNunchakuAndLoRAs
from pathlib import Path

application_path = Path(__file__).resolve().parents[2]

def test_FluxDepthNunchakuAndLoRAs_create_control_image():

    application_paths = ApplicationPaths.create(
        configpath=application_path)

    terminal_ui = TerminalUI()

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)

    process_configurations.process_configurations()

    flux_depth_nunchaku_and_loras = FluxDepthNunchakuAndLoRAs(
        process_configurations.configurations["nunchaku_flux_control_configuration"],
        process_configurations.configurations["flux_generation_configuration"],
        process_configurations.configurations["pipeline_inputs"],
        process_configurations.configurations["nunchaku_loras_configuration"])

    assert flux_depth_nunchaku_and_loras._control_images == []

    flux_depth_nunchaku_and_loras.create_control_image()

    assert flux_depth_nunchaku_and_loras._control_images is not None

def test_steps_of_generate_depth_image_explicitly():
    """
    We explicitly run each step of the function generate_depth_image() in
    GenerateImages.
    """

    application_paths = ApplicationPaths.create(
        configpath=application_path)

    terminal_ui = TerminalUI()

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)

    process_configurations.process_configurations()

    flux_depth_nunchaku_and_loras = FluxDepthNunchakuAndLoRAs(
        process_configurations.configurations["nunchaku_flux_control_configuration"],
        process_configurations.configurations["flux_generation_configuration"],
        process_configurations.configurations["pipeline_inputs"],
        process_configurations.configurations["nunchaku_loras_configuration"])

    assert flux_depth_nunchaku_and_loras._processor_enabled is False

    flux_depth_nunchaku_and_loras.create_control_image()

    assert flux_depth_nunchaku_and_loras._processor_enabled is True

    flux_depth_nunchaku_and_loras._delete_processor()

    assert flux_depth_nunchaku_and_loras._processor_enabled is False

    assert flux_depth_nunchaku_and_loras._prompt_embeds == []
    assert flux_depth_nunchaku_and_loras._text_encoder_2_enabled is False

    flux_depth_nunchaku_and_loras.create_prompt_embeds()

    assert flux_depth_nunchaku_and_loras._prompt_embeds is not None
    assert flux_depth_nunchaku_and_loras._text_encoder_2_enabled

    flux_depth_nunchaku_and_loras._delete_text_encoder_2_and_pipeline()

    assert flux_depth_nunchaku_and_loras._text_encoder_2_enabled is False

    flux_depth_nunchaku_and_loras.create_transformer_and_pipeline()
    assert flux_depth_nunchaku_and_loras._transformer_enabled is True

    flux_depth_nunchaku_and_loras.update_transformer_with_loras()

    images = flux_depth_nunchaku_and_loras.call_pipeline(
        0,
        0)

    assert images is not None
    print(type(images))
    print(images)

    batch_processing_configuration = \
        process_configurations.get_batch_processing_configuration()

    batch_processing_configuration.create_and_save_image(
        0,
        images[0],
        flux_depth_nunchaku_and_loras._generation_configuration,
        process_configurations.get_model_name())

    assert True


