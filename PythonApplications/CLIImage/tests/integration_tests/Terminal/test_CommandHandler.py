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
                "nunchaku_loras_configuration": configuration,
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
