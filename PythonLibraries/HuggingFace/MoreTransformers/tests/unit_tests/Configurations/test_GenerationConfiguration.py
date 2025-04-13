from pathlib import Path

from moretransformers.Configurations import GenerationConfiguration

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerationConfiguration_loads_from_blank_fields():
    generation_configuration = GenerationConfiguration(
        test_data_directory / "blank_generation_configuration.yml")

    assert generation_configuration.configuration_path == test_data_directory / "blank_generation_configuration.yml"
    assert generation_configuration.timeout == 60.0
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