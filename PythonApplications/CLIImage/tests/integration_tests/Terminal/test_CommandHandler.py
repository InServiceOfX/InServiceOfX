from types import SimpleNamespace

from cliimage.Terminal.CommandHandler import CommandHandler
from morediffusers.Configurations.NunchakuLoRAsConfiguration import (
    NunchakuLoRAsConfiguration)


class RecordingTerminalUI:
    def __init__(self):
        self.messages = []

    def print_info(self, message):
        self.messages.append(("info", message))

    def print_success(self, message):
        self.messages.append(("success", message))

    def print_error(self, message):
        self.messages.append(("error", message))

    def print_goodbye(self):
        self.messages.append(("goodbye", ""))

    def print_help(self, message):
        self.messages.append(("help", message))


def _write_loras_configuration(path):
    path.write_text(
        "\n".join([
            "lora_scale: 0.5",
            "loras:",
            "  - nickname: hero style",
            "    directory_path: /Data1/Models/Diffusion/LoRAs",
            "    filename: hero.safetensors",
            "    lora_strength: 0.9",
            "    is_active: true",
            "  - nickname: inactive style",
            "    directory_path: /Data1/Models/Diffusion/LoRAs",
            "    filename: inactive.safetensors",
            "    lora_strength: 0.4",
            "    is_active: false",
            "",
        ]))


def _create_handler(config_path):
    configuration = NunchakuLoRAsConfiguration.from_yaml(config_path)
    app = SimpleNamespace(
        _application_paths=SimpleNamespace(
            configuration_file_paths={
                "nunchaku_loras_configuration": config_path,
            }),
        _process_configurations=SimpleNamespace(
            configurations={
                "batch_processing_configuration": SimpleNamespace(
                    number_of_images=3),
                "flux_generation_configuration": SimpleNamespace(
                    width=1024,
                    height=768,
                    num_inference_steps=20,
                    guidance_scale=2.5,
                    true_cfg_scale=1.0,
                    temporary_save_path="/Data/Private"),
                "nunchaku_configuration": SimpleNamespace(
                    cuda_device="cuda:0",
                    flux_model_path="/Data/Models/FLUX.1-dev",
                    nunchaku_model_paths=[
                        "/Data/Models/nunchaku-model.safetensors",
                    ]),
                "nunchaku_loras_configuration": configuration,
                "pipeline_inputs": SimpleNamespace(
                    prompt="a short test prompt",
                    negative_prompt="ugly"),
            }),
        _terminal_ui=RecordingTerminalUI())

    return CommandHandler(app)


def test_CommandHandler_updates_lora_yaml_by_nickname(tmp_path):
    config_path = tmp_path / "nunchaku_loras_configuration.yml"
    _write_loras_configuration(config_path)
    handler = _create_handler(config_path)

    continue_running, handled = handler.handle_command(
        '.enable_lora "inactive style"')

    assert continue_running is True
    assert handled is True

    continue_running, handled = handler.handle_command(
        '.set_lora_strength "inactive style" 1.25')

    assert continue_running is True
    assert handled is True

    reloaded = NunchakuLoRAsConfiguration.from_yaml(config_path)

    assert reloaded.loras["inactive style"].is_active is True
    assert reloaded.loras["inactive style"].lora_strength == 1.25
    assert reloaded.loras["hero style"].is_active is True


def test_CommandHandler_status_reports_current_configuration(tmp_path):
    config_path = tmp_path / "nunchaku_loras_configuration.yml"
    _write_loras_configuration(config_path)
    handler = _create_handler(config_path)

    continue_running, handled = handler.handle_command(".status")

    assert continue_running is True
    assert handled is True
    messages = handler._app._terminal_ui.messages
    status_messages = [
        message for message_type, message in messages
        if message_type == "info" and "CLIImage status:" in message
    ]
    assert status_messages
    assert "Nunchaku models: 1" in status_messages[0]
    assert "Active LoRAs: 1" in status_messages[0]
