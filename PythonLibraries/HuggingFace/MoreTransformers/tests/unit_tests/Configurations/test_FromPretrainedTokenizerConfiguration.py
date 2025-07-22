from moretransformers.Configurations import FromPretrainedTokenizerConfiguration
from corecode.Utilities import DataSubdirectories

data_subdirectories = DataSubdirectories()

model_path = data_subdirectories.Data / "Models" / "LLM" / "HuggingFaceTB" / \
    "SmolLM3-3B"

def test_FromPretrainedTokenizerConfiguration_instantiates():
    config = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)
    assert config is not None
    assert isinstance(config, FromPretrainedTokenizerConfiguration)
    assert config.pretrained_model_name_or_path == model_path
    assert config.local_files_only == True
    assert config.force_download == None
    assert config.trust_remote_code == True

def test_to_dict_works():
    config = FromPretrainedTokenizerConfiguration(
        pretrained_model_name_or_path=model_path)

    config_dict = config.to_dict()
    assert config_dict is not None
    assert isinstance(config_dict, dict)
    assert config_dict["pretrained_model_name_or_path"] == model_path
    assert config_dict["local_files_only"] == True
    assert config_dict["trust_remote_code"] == True

    assert set(config_dict.keys()) == {
        "pretrained_model_name_or_path",
        "local_files_only",
        "trust_remote_code"}
