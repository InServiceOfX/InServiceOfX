from pathlib import Path

from moretransformers.Configurations import Configuration

import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_Configuration_parses():
    configuration = Configuration(test_data_directory / "configuration-llama3.yml")
    assert configuration.model_path == \
        "/Data/Models/LLM/meta-llama/Llama-3.2-1B-Instruct"
    assert configuration.model_name == "Llama-3.2-1B-Instruct"
    assert configuration.torch_dtype == torch.float16
    assert configuration.task == "text-generation"

    