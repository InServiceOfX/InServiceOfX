"""Lightweight tests for Qwen3VLConfiguration; no vLLM/torch needed.

Run with: pytest -q from the MoreMinerU directory after `pip install pydantic
pyyaml pytest`. Inside the VLLMMultimodal container these are already present.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moremineru.Configurations import Qwen3VLConfiguration  # noqa: E402


VALID_YAML = """\
model_path: /tmp/does-not-exist/Qwen3-VL-4B-Instruct
system_prompt: You are a precise extraction assistant.
vllm_engine_kwargs:
  dtype: bfloat16
  max_model_len: 8192
  gpu_memory_utilization: 0.85
  enforce_eager: true
default_sampling_params:
  max_tokens: 2048
  temperature: 0.0
"""


def test_from_yaml_loads_required_fields(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(VALID_YAML)

    config = Qwen3VLConfiguration.from_yaml(config_path)

    assert config.model_path == Path("/tmp/does-not-exist/Qwen3-VL-4B-Instruct")
    assert config.system_prompt == "You are a precise extraction assistant."
    assert config.vllm_engine_kwargs["dtype"] == "bfloat16"
    assert config.default_sampling_params["max_tokens"] == 2048
    assert config.get_tensor_parallel_size() == 1


def test_from_yaml_missing_required_field_raises(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text("system_prompt: hello\n")

    with pytest.raises(ValueError) as excinfo:
        Qwen3VLConfiguration.from_yaml(config_path)
    assert "model_path" in str(excinfo.value)


def test_validate_paths_rejects_missing_path(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(VALID_YAML)

    with pytest.raises(ValueError) as excinfo:
        Qwen3VLConfiguration.from_yaml(config_path, validate_paths=True)
    assert "does not exist" in str(excinfo.value)


def test_default_sampling_params_when_omitted(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(
        "model_path: /tmp/does-not-exist/Qwen3-VL-4B-Instruct\n"
    )

    config = Qwen3VLConfiguration.from_yaml(config_path)
    # Field defaults match the Qwen3-VL model card's recommended VL-task
    # generation hyperparameters (not greedy — see Qwen3VLConfiguration).
    assert config.default_sampling_params["temperature"] == 0.7
    assert config.default_sampling_params["top_p"] == 0.8
    assert config.default_sampling_params["top_k"] == 20
    assert config.default_sampling_params["presence_penalty"] == 1.5
    assert config.default_sampling_params["max_tokens"] == 1024


def test_rejects_minerU_specific_keys_in_engine_kwargs(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(
        "model_path: /tmp/does-not-exist/Qwen3-VL-4B-Instruct\n"
        "vllm_engine_kwargs:\n"
        "  dtype: bfloat16\n"
        "  image_analysis: true\n"
    )

    with pytest.raises(ValueError) as excinfo:
        Qwen3VLConfiguration.from_yaml(config_path)
    assert "image_analysis" in str(excinfo.value)


def test_image_pixel_caps_have_defaults(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(
        "model_path: /tmp/does-not-exist/Qwen3-VL-4B-Instruct\n"
    )

    config = Qwen3VLConfiguration.from_yaml(config_path)
    # Defaults cap visual-token count comfortably under max_model_len=8192:
    # 1280 * 28 * 28 = 1,003,520 pixels (~1280 patches)
    # 256  * 28 * 28 =   200,704 pixels (~256 patches)
    assert config.image_max_pixels == 1280 * 28 * 28
    assert config.image_min_pixels == 256 * 28 * 28


def test_image_pixel_caps_can_be_disabled(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(
        "model_path: /tmp/does-not-exist/Qwen3-VL-4B-Instruct\n"
        "image_max_pixels: null\n"
        "image_min_pixels: null\n"
    )

    config = Qwen3VLConfiguration.from_yaml(config_path)
    assert config.image_max_pixels is None
    assert config.image_min_pixels is None


def test_save_yaml_round_trip(tmp_path: Path):
    config_path = tmp_path / "qwen.yml"
    config_path.write_text(VALID_YAML)
    config = Qwen3VLConfiguration.from_yaml(config_path)

    out_path = tmp_path / "out" / "qwen.yml"
    config.save_yaml(out_path)
    reloaded = Qwen3VLConfiguration.from_yaml(out_path)

    assert reloaded.model_path == config.model_path
    assert reloaded.system_prompt == config.system_prompt
    assert reloaded.vllm_engine_kwargs == config.vllm_engine_kwargs
    assert reloaded.default_sampling_params == config.default_sampling_params
