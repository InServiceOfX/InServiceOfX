from moretransformers.Applications import ModelAndTokenizer
from moretransformers.Configurations import FromPretrainedModelConfiguration
from corecode.Utilities import DataSubdirectories

data_subdirectories = DataSubdirectories()

model_path = data_subdirectories.Data / "Models" / "LLM" / "HuggingFaceTB" / \
    "SmolLM3-3B"

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

def test_ModelAndTokenizer_instantiates_with_model_configuration():

    config = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path)

    model_and_tokenizer = ModelAndTokenizer(
        model_path=model_path,
        from_pretrained_model_configuration=config)
