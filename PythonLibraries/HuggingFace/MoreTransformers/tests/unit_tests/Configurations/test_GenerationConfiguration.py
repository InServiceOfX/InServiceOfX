from pathlib import Path

from moretransformers.Configurations import GenerationConfiguration

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerationConfiguration_loads_from_blank_fields():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "blank_generation_configuration.yml")

    assert generation_configuration.max_new_tokens is None
    assert generation_configuration.do_sample is None
    assert generation_configuration.use_cache is None
    assert generation_configuration.temperature is None
    assert generation_configuration.top_k is None
    assert generation_configuration.top_p is None
    assert generation_configuration.repetition_penalty is None
    assert generation_configuration.eos_token_id is None
    assert generation_configuration.pad_token_id is None

    config_as_dict = generation_configuration.to_dict()
    assert len(config_as_dict.keys()) == 0

def test_GenerationConfiguration_inits():
    generation_configuration = GenerationConfiguration()

    config_as_dict = generation_configuration.to_dict()
    assert len(config_as_dict.keys()) == 1
    assert config_as_dict == {"do_sample": False}

    # None values are not included in the dictionary
    generation_configuration.do_sample = None
    config_as_dict = generation_configuration.to_dict()
    assert len(config_as_dict.keys()) == 0
    assert config_as_dict == {}

    generation_configuration.do_sample = True
    generation_configuration.temperature = 0.9
    generation_configuration.min_p=0.15
    generation_configuration.repetition_penalty=1.05
    generation_configuration.max_new_tokens=65536

    config_as_dict = generation_configuration.to_dict()
    assert len(config_as_dict.keys()) == 5
    assert set(config_as_dict.keys()) == {
        "do_sample",
        "temperature",
        "min_p",
        "repetition_penalty",
        "max_new_tokens"
    }

    assert config_as_dict["do_sample"] == True
    assert config_as_dict["temperature"] == 0.9
