import pytest
from pathlib import Path

from safetensors.torch import safe_open

def is_directory_empty_or_missing(directory_path: str) -> bool:
    path = Path(directory_path)
    return not path.exists() or not any(path.iterdir())

MODEL_DIR = "/Data/Models/LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_safetensors_conversion():
    paths = [file_path for file_path in Path(MODEL_DIR).glob("*.safetensors")]
    assert len(paths) == 1
    assert "model.safetensors" in paths[0].name

    f = safe_open(paths[0], framework="pt", device="cpu")

    assert type(f) == safe_open
    safe_open_keys = f.keys()

    assert len(safe_open_keys) == 339

    keys_that_start_with_model = [key for key in safe_open_keys
        if key.startswith("model.")]
    assert len(keys_that_start_with_model) == 338

    keys_that_start_with_model = [key[len("model."):]
        for key in keys_that_start_with_model]

    print(keys_that_start_with_model)

    other_keys = [key for key in safe_open_keys
        if not key.startswith("model.")]
    assert len(other_keys) == 1
    assert other_keys[0] == "lm_head.weight"

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_another_safetensors_test():
    # Another test
    assert True
