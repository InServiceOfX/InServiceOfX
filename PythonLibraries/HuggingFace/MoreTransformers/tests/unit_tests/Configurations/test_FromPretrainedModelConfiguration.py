from moretransformers.Configurations import FromPretrainedModelConfiguration
from corecode.Utilities import DataSubdirectories

data_subdirectories = DataSubdirectories()

model_path = data_subdirectories.Data / "Models" / "LLM" / "HuggingFaceTB" / \
    "SmolLM3-3B"

def test_FromPretrainedModelConfiguration_instantiates():
    config = FromPretrainedModelConfiguration(
        pretrained_model_name_or_path=model_path)
    assert config is not None
    assert isinstance(config, FromPretrainedModelConfiguration)
    assert config.pretrained_model_name_or_path == model_path
    assert config.local_files_only == True
    assert config.force_download == None
    assert config.use_safetensors == None
    assert config.trust_remote_code == True
    assert config.attn_implementation == None
    assert config.torch_dtype == None
    assert config.device_map == None

def test_to_dict_works():
    config = FromPretrainedModelConfiguration(
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

    config.use_safetensors = True
    config.attn_implementation = "flash_attention_2"
    config.device_map = "cuda:0"

    config_dict = config.to_dict()
    assert config_dict is not None
    assert isinstance(config_dict, dict)
    assert config_dict["use_safetensors"] == True
    assert config_dict["attn_implementation"] == "flash_attention_2"
    assert config_dict["device_map"] == "cuda:0"

    assert set(config_dict.keys()) == {
        "pretrained_model_name_or_path",
        "local_files_only",
        "trust_remote_code",
        "use_safetensors",
        "attn_implementation",
        "device_map"}

