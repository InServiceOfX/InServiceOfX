from corecode.Utilities import DataSubdirectories

from transformers import PreTrainedTokenizerFast

data_sub_dirs = DataSubdirectories()

def test_PreTrainedTokenizerFast_instantiates():

    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_path)

    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # ID of beginning of sentence token in the vocabulary.
    assert tokenizer.bos_token_id == 128000
    # ID of end of sentence token in vocabulary.
    assert tokenizer.eos_token_id == 128009
    # ID of padding token in the vocabulary.
    assert tokenizer.pad_token_id == None
