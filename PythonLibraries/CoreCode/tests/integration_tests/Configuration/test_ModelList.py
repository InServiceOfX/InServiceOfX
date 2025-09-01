from pathlib import Path
from corecode.Configuration import ModelList
from copy import deepcopy

test_data_path = Path(__file__).parents[2] / "TestData"

def test_model_list_from_yaml():
    model_list = ModelList.from_yaml(test_data_path / "model_list.yml")
    assert model_list is not None
    assert len(model_list) == 5
    assert "qwen3-0.6b" in model_list
    assert "gemma-3-1b-it" in model_list
    assert "gemma-3-270m-it" in model_list

    original_list = deepcopy(model_list.models)
    assert original_list == model_list.models

    assert original_list["qwen3-0.6b"] == \
        Path("/Data/Models/LLM/Qwen/Qwen3-0.6B")
    assert original_list["gemma-3-1b-it"] == \
        Path("/Data/Models/LLM/google/gemma-3-1b-it")
    assert original_list["gemma-3-270m-it"] == \
        Path("/Data/Models/LLM/google/gemma-3-270m-it")
    assert original_list["lfm2-1.2b"] == \
        Path("/Data/Models/LLM/LiquidAI/LFM2-1.2B")
    assert original_list["lucy-128k"] == \
        Path("/Data/Models/LLM/Menlo/Lucy-128k")

def test_model_list_to_yaml_works():
    model_list = ModelList.from_yaml(test_data_path / "model_list.yml")
    original_list = deepcopy(model_list.models)
    model_list.add_model(
        "new_model",
        "/Data1/Models/LLM/HuggingFaceTB/SmolLM2-360M-Instruct")
    changed_list = deepcopy(model_list.models)
    model_list.to_yaml(test_data_path / "model_list.yml")
    model_list = ModelList.from_yaml(test_data_path / "model_list.yml")
    assert model_list.models == changed_list
    model_list.remove_model("new_model")
    assert model_list.models == original_list
    model_list.to_yaml(test_data_path / "model_list.yml")
    model_list = ModelList.from_yaml(test_data_path / "model_list.yml")
    assert model_list.models == original_list
