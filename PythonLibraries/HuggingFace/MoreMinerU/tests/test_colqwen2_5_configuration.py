"""Lightweight tests for ColQwen2_5Configuration; no torch/colpali needed.

Run with: pytest -q from the MoreMinerU directory after `pip install pydantic
pyyaml pytest`. Inside the VLLMMultimodal container these are already present.
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moremineru.Configurations import ColQwen2_5Configuration  # noqa: E402


VALID_YAML = """\
model_path: /tmp/does-not-exist/colqwen2.5-v0.2
torch_dtype: bfloat16
device_map: cuda:0
attn_implementation: null
"""


def test_from_yaml_loads_required_fields(tmp_path: Path):
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text(VALID_YAML)

    config = ColQwen2_5Configuration.from_yaml(config_path)

    assert config.model_path == Path("/tmp/does-not-exist/colqwen2.5-v0.2")
    assert config.torch_dtype == "bfloat16"
    assert config.device_map == "cuda:0"
    assert config.attn_implementation is None


def test_from_yaml_missing_required_field_raises(tmp_path: Path):
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text("torch_dtype: bfloat16\n")

    with pytest.raises(ValueError) as excinfo:
        ColQwen2_5Configuration.from_yaml(config_path)
    assert "model_path" in str(excinfo.value)


def test_validate_paths_rejects_missing_model_path(tmp_path: Path):
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text(VALID_YAML)

    with pytest.raises(ValueError) as excinfo:
        ColQwen2_5Configuration.from_yaml(config_path, validate_paths=True)
    assert "does not exist" in str(excinfo.value)


def test_validate_paths_rejects_missing_adapter_config(tmp_path: Path):
    # model_path exists but lacks adapter_config.json — the wrapper should
    # surface the LoRA-expected layout instead of the user discovering a
    # cryptic error from colpali-engine.
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text(
        f"model_path: {model_dir}\n"
        "torch_dtype: bfloat16\n"
        "device_map: cuda:0\n"
    )

    with pytest.raises(ValueError) as excinfo:
        ColQwen2_5Configuration.from_yaml(config_path, validate_paths=True)
    assert "adapter_config.json" in str(excinfo.value)


def test_save_yaml_round_trip(tmp_path: Path):
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text(VALID_YAML)
    config = ColQwen2_5Configuration.from_yaml(config_path)

    out_path = tmp_path / "out" / "colqwen.yml"
    config.save_yaml(out_path)
    reloaded = ColQwen2_5Configuration.from_yaml(out_path)

    assert reloaded.model_path == config.model_path
    assert reloaded.torch_dtype == config.torch_dtype
    assert reloaded.device_map == config.device_map
    assert reloaded.attn_implementation == config.attn_implementation


def test_default_attn_implementation_none(tmp_path: Path):
    config_path = tmp_path / "colqwen.yml"
    config_path.write_text(
        "model_path: /tmp/does-not-exist/colqwen2.5-v0.2\n"
    )
    config = ColQwen2_5Configuration.from_yaml(config_path)
    # Default leaves attn picking to transformers (SDPA on modern GPUs);
    # the VLLMMultimodal image has flash-attn uninstalled so don't default
    # to flash_attention_2.
    assert config.attn_implementation is None
    assert config.torch_dtype == "bfloat16"
    assert config.device_map == "cuda:0"
