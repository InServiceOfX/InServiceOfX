from pathlib import Path

from morediffusers.Configurations.NunchakuLoRAsConfiguration import (
    NunchakuLoRAsConfiguration)


def test_NunchakuLoRAsConfiguration_save_yaml_preserves_editable_schema(
        tmp_path):
    config_path = tmp_path / "nunchaku_loras_configuration.yml"
    config_path.write_text(
        "\n".join([
            "lora_scale: 0.5",
            "loras:",
            "  - nickname: hero",
            "    directory_path: /Data1/Models/Diffusion/LoRAs",
            "    filename: hero.safetensors",
            "    lora_strength: 0.9",
            "    is_active: true",
            "  - nickname: inactive",
            "    directory_path: /Data1/Models/Diffusion/LoRAs",
            "    filename: inactive.safetensors",
            "    lora_strength: 0.4",
            "    is_active: false",
            "",
        ]))

    configuration = NunchakuLoRAsConfiguration.from_yaml(config_path)

    assert len(configuration.loras) == 2
    assert list(configuration.get_active_loras().keys()) == ["hero"]

    configuration.toggle_lora("inactive")
    configuration.set_lora_strength("hero", 1.1)
    configuration.save_yaml(config_path)

    reloaded = NunchakuLoRAsConfiguration.from_yaml(config_path)

    assert reloaded.loras["inactive"].is_active is True
    assert reloaded.loras["hero"].lora_strength == 1.1
    assert isinstance(reloaded.loras["hero"].directory_path, Path)
    saved_text = config_path.read_text()
    assert "!!python" not in saved_text
    assert "directory_path: /Data1/Models/Diffusion/LoRAs" in saved_text
