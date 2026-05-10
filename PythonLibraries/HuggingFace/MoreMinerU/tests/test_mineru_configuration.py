"""Lightweight tests that don't need vLLM/torch installed.

Run with: pytest -q from the MoreMinerU directory after `pip install pydantic
pyyaml pytest`. Inside the VLLMMultimodal container these are already present.
"""
from pathlib import Path

import pytest

import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from moremineru.Configurations import MinerUConfiguration  # noqa: E402


VALID_YAML = """\
model_path: /tmp/does-not-exist/MinerU2.5-Pro
backend: vllm-engine
image_analysis: false
vllm_engine_kwargs:
  dtype: bfloat16
  max_model_len: 8192
  gpu_memory_utilization: 0.85
  enforce_eager: true
"""


def test_from_yaml_loads_required_fields(tmp_path: Path):
    config_path = tmp_path / "mineru.yml"
    config_path.write_text(VALID_YAML)

    config = MinerUConfiguration.from_yaml(config_path)

    assert config.model_path == Path("/tmp/does-not-exist/MinerU2.5-Pro")
    assert config.backend == "vllm-engine"
    assert config.image_analysis is False
    assert config.vllm_engine_kwargs["dtype"] == "bfloat16"
    assert config.get_tensor_parallel_size() == 1


def test_from_yaml_missing_required_field_raises(tmp_path: Path):
    config_path = tmp_path / "mineru.yml"
    config_path.write_text("backend: vllm-engine\n")

    with pytest.raises(ValueError) as excinfo:
        MinerUConfiguration.from_yaml(config_path)
    assert "model_path" in str(excinfo.value)


def test_validate_paths_rejects_missing_path(tmp_path: Path):
    config_path = tmp_path / "mineru.yml"
    config_path.write_text(VALID_YAML)

    with pytest.raises(ValueError) as excinfo:
        MinerUConfiguration.from_yaml(config_path, validate_paths=True)
    assert "does not exist" in str(excinfo.value)


def test_unsupported_backend_rejected(tmp_path: Path):
    config_path = tmp_path / "mineru.yml"
    config_path.write_text(
        VALID_YAML.replace("backend: vllm-engine", "backend: transformers")
    )

    with pytest.raises(ValueError) as excinfo:
        MinerUConfiguration.from_yaml(config_path)
    assert "vllm-engine" in str(excinfo.value)


def test_save_yaml_round_trip(tmp_path: Path):
    config_path = tmp_path / "mineru.yml"
    config_path.write_text(VALID_YAML)
    config = MinerUConfiguration.from_yaml(config_path)

    out_path = tmp_path / "out" / "mineru.yml"
    config.save_yaml(out_path)
    reloaded = MinerUConfiguration.from_yaml(out_path)

    assert reloaded.model_path == config.model_path
    assert reloaded.vllm_engine_kwargs == config.vllm_engine_kwargs
