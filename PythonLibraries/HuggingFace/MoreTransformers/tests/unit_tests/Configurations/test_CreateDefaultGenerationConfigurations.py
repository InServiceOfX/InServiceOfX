from moretransformers.Configurations import (
    CreateDefaultGenerationConfigurations,
    GenerationConfiguration)

def test_for_Qwen3_thinking():
    generation_configuration = \
        CreateDefaultGenerationConfigurations.for_Qwen3_thinking()
    assert generation_configuration is not None
    assert isinstance(generation_configuration, GenerationConfiguration)
    assert generation_configuration.do_sample == False
    assert generation_configuration.temperature == 0.6
    assert generation_configuration.top_p == 0.95
    assert generation_configuration.top_k == 20
    assert generation_configuration.min_p == 0.0
    assert generation_configuration.repetition_penalty == None
    assert generation_configuration.eos_token_id == None
    assert generation_configuration.pad_token_id == None
    assert generation_configuration.use_cache == None
    assert generation_configuration.max_new_tokens == 32768

    generation_dict = generation_configuration.to_dict()
    assert generation_dict is not None
    assert isinstance(generation_dict, dict)
    assert generation_dict["do_sample"] == False
    assert generation_dict["temperature"] == 0.6
    assert generation_dict["top_p"] == 0.95
    assert generation_dict["max_new_tokens"] == 32768
    assert generation_dict["top_k"] == 20
    assert generation_dict["min_p"] == 0.0

    generation_configuration.do_sample = True

    generation_dict = generation_configuration.to_dict()
    assert generation_dict is not None
    assert isinstance(generation_dict, dict)
    assert generation_dict["do_sample"] == True
    assert generation_dict["temperature"] == 0.6
    assert generation_dict["top_p"] == 0.95
    assert generation_dict["max_new_tokens"] == 32768
    assert generation_dict["top_k"] == 20
    assert generation_dict["min_p"] == 0.0

    print("generation_dict: ", generation_dict)
