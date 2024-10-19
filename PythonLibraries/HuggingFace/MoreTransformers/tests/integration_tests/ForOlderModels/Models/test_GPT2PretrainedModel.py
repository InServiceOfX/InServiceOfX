from moretransformers.Configurations import Configuration
from transformers import (modeling_utils, GPT2PreTrainedModel, GPT2Config)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME)

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_gpt2 = Configuration(
    test_data_directory / "configuration-gpt2.yml")

def test__add_variant_creates_filenames():
    assert modeling_utils._add_variant(SAFE_WEIGHTS_NAME, "variant") == \
        "model.variant.safetensors"
    assert modeling_utils._add_variant(SAFE_WEIGHTS_INDEX_NAME, "variant") == \
        "model.safetensors.index.variant.json"
    assert modeling_utils._add_variant(WEIGHTS_INDEX_NAME, "variant") == \
        "pytorch_model.bin.index.variant.json"
    assert modeling_utils._add_variant(WEIGHTS_NAME, "variant") == \
        "pytorch_model.variant.bin"


def test_GPT2PreTrainedModel_instantiates_from_pretrained():
    """
    See modeling_utils.py for PretrainedModel class and def __init__() and
    def post_init(). 
    """
    # It should be of type GPT2Config, but it fails.
    #assert isinstance(GPT2PreTrainedModel.config_class, GPT2Config)

    # Parameters
    # variant ('str', *optional*):
    #   If specified load weights from 'variant' filename, *e.g.*
    # pytorch_model.<variant>.bin. 'variant' is ignored when using 'from_tf',
    # 'from_flax'.
    model = GPT2PreTrainedModel.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)
    
