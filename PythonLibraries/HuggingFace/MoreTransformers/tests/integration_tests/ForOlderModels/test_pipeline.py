from moretransformers.Configurations import Configuration

from transformers import (
    GPT2LMHeadModel,
    GPT2PreTrainedModel,
    PreTrainedTokenizerFast)

from transformers import pipeline, set_seed, Pipeline, GenerationConfig

from pathlib import Path
import json
import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"
configuration_gpt2 = Configuration(
    test_data_directory / "configuration-gpt2.yml")

def test_configuration_works():

    assert isinstance(configuration_gpt2, Configuration)
    assert configuration_gpt2.task == "text-generation"
    assert "gpt2" in configuration_gpt2.model_path
    assert "openai-community" in configuration_gpt2.model_path
    assert configuration_gpt2.torch_dtype == ""

def test_pipeline_instantiates():
    """
    In src/transformers/pipelines/__init__.py, the pipeline function is defined.
    def pipeline(
        task: str,
        model,
        Optional[Union[str, "PreTrainedModel", ..]],
        ..) -> Pipeline
    It's a utility factory method to build a ['Pipeline]
    "text-generation" will return a ['TextGenerationPipeline']

    See pipelines/base.py for class Pipeline.
    """
    assert Path(configuration_gpt2.model_path).exists()

    pipeline_object = pipeline(
        task="text-generation",
        model=configuration_gpt2.model_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    assert isinstance(pipeline_object, Pipeline)
    assert pipeline_object.tokenizer is not None
    assert isinstance(pipeline_object.tokenizer, PreTrainedTokenizerFast)
    assert pipeline_object.device == torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    assert pipeline_object.binary_output is False
    assert pipeline_object.framework == "pt"
    # From pipelines/base.py, class Pipeline, def __init__(..), if model can
    # generate, create local generation config; this is done to avoid side
    # effects on model as we apply local tweaks to generation config. 
    assert pipeline_object.model.can_generate()

    assert isinstance(pipeline_object.model, GPT2LMHeadModel)
    assert isinstance(pipeline_object.model, GPT2PreTrainedModel)

    try:
        assert pipeline_object.generation_config is not None
        assert isinstance(pipeline_object.generation_config, GenerationConfig)
        assert pipeline_object.generation_config["bos_taken_id"] == 50256
        assert pipeline_object.generation_config["do_sample"]
        assert pipeline_object.generation_config["eos_token_id"] == 50256
        assert pipeline_object.generation_config["max_length"] == 50
    except TypeError as err:
        if str(err) == "'GenerationConfig' object is not subscriptable":
            generation_config_as_json = \
                json.loads(pipeline_object.generation_config.to_json_string())
            print(generation_config_as_json.keys())
            assert generation_config_as_json["bos_token_id"] == 50256
            assert generation_config_as_json["do_sample"]
            assert generation_config_as_json["eos_token_id"] == 50256
            assert generation_config_as_json["max_length"] == 50
        else:
            pytest.fail(f'Unexpected error: {err}')


def test_pipeline_calls():
    pipeline_object = pipeline(
        task="text-generation",
        model=configuration_gpt2.model_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    set_seed(42)

    # From pipelines/text_generation.py, class TextGenerationPipeline,
    # def __call__(self, text_inputs, **kwargs), completes the prompt(s) given
    # as inputs.
    # Args:
    #     text_inputs ('str', 'List[str]', List[Dict[str, str]],..):
    #         If strings or list of strings are passed, this pipeline will
    #         continue each prompt. Alternatively, a "chat", in form of a list
    #         of dicts with "role" and "content" keys, can be passed.
    # TextGenerationPipeline calls super().__call__(Chat(text_inputs), **kwargs)
    # See text_generation.py for class Chat, which only checks for "role" and
    # "content" keys. and code comment in the beginning of
    # TextGenerationPipeline, where example of "role" includes "user" and
    # "assistant".

    output = pipeline_object(
        "Hello, I'm a language model,",
        max_length=30,
        num_return_sequences=5)
    assert isinstance(output, list)
    assert len(output) == 5
    assert isinstance(output[0], dict)
    assert len(output[0].keys()) == 1
    assert len(output[0]["generated_text"]) == 132