from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import FromPretrainedModelConfiguration
from corecode.Utilities import DataSubdirectories

data_subdirectories = DataSubdirectories()

model_path = data_subdirectories.Data / "Models" / "LLM" / "HuggingFaceTB" / \
    "SmolLM3-3B"

import torch

def test_ModelAndTokenizer_instantiates():
    mat = ModelAndTokenizer(model_path=model_path)

    assert mat._model_path == model_path

    assert mat._fpmc.to_dict() == {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": True,
        "trust_remote_code": True}

    assert mat._fptc.to_dict() == {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": True,
        "trust_remote_code": True}
    
    assert mat._generation_configuration.to_dict() == {
        "max_new_tokens": 100,
        "do_sample": False}

def test_ModelAndTokenizer_can_modify_configurations():
    mat = ModelAndTokenizer(model_path=model_path)

    mat._fpmc.device_map = "cuda:0"
    mat._fpmc.torch_dtype = torch.bfloat16

    assert mat._fpmc.to_dict() == {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": True,
        "trust_remote_code": True,
        "device_map": "cuda:0",
        "torch_dtype": torch.bfloat16}

    mat._generation_configuration.max_new_tokens = 65536
    mat._generation_configuration.do_sample = True
    mat._generation_configuration.temperature = 0.3
    mat._generation_configuration.min_p = 0.15
    mat._generation_configuration.repetition_penalty = 1.05

    assert mat._generation_configuration.to_dict() == {
        "max_new_tokens": 65536,
        "do_sample": True,
        "temperature": 0.3,
        "min_p": 0.15,
        "repetition_penalty": 1.05}

def test_ModelAndTokenizer_instantiates_with_model_configuration():

    config = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path)

    model_and_tokenizer = ModelAndTokenizer(
        model_path=model_path,
        from_pretrained_model_configuration=config)
