from moretransformers.Configurations import Configuration
from transformers import GPT2Tokenizer

from pathlib import Path

import torch

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"

configuration_gpt2 = Configuration(test_data_directory / "configuration-gpt2.yml")

def test_GPT2Tokenizer_instantiates():
    """
    See models/gpt2/tokenization_gpt2.py for
    class GPT2Tokenizer(PreTrainedTokenizer). Then see tokenization_utils.py for
    class PreTrainedTokenizer(PreTrainedTokenizerBase). Then see
    tokenization_utils_base.py for
    class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin). Only in
    this base class is def from_pretrained(..) defined.

    In from_pretrained(..), it calls load_from_pretrained(..), which effectively
    calls def __init__(..).
    """

    tokenizer = GPT2Tokenizer.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    assert isinstance(tokenizer, GPT2Tokenizer)

    assert tokenizer.add_bos_token == False
    assert isinstance(tokenizer.encoder, dict)
    assert len(tokenizer.encoder.keys()) == 50257
    assert isinstance(tokenizer.decoder, dict)
    assert len(tokenizer.decoder.keys()) == 50257
    assert isinstance(tokenizer.byte_decoder, dict)
    assert len(tokenizer.byte_decoder.keys()) == 256
    assert isinstance(tokenizer.bpe_ranks, dict)
    assert len(tokenizer.bpe_ranks.keys()) == 50000
    assert ('h', 'e') in tokenizer.bpe_ranks.keys()
    assert ('i', 'n') in tokenizer.bpe_ranks.keys()
    assert tokenizer.cache == {}
    assert tokenizer.add_prefix_space == False

def test_GPT2Tokenizer_calls():

    text = "Replace me by any text you'd like."

    tokenizer = GPT2Tokenizer.from_pretrained(
        configuration_gpt2.model_path,
        local_files_only=True)

    encoded_input = tokenizer(text, return_tensors="pt")

    assert len(encoded_input.keys()) == 2
    assert isinstance(encoded_input["input_ids"], torch.Tensor)
    assert encoded_input["input_ids"].shape == torch.Size([1, 10])
    assert isinstance(encoded_input["attention_mask"], torch.Tensor)
    assert encoded_input["attention_mask"].shape == torch.Size([1, 10])