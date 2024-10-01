from corecode.Utilities import (
    DataSubdirectories,
    )

from transformers import AutoTokenizer, TextIteratorStreamer

data_sub_dirs = DataSubdirectories()

def test_TextIteratorStreamer_instantiates():
    """
    See generation/streamers.py for class TextIteratorStreamer(TextStreamer).

    Parameters:
        skip_prompt - whether to skip the prompt to '.generate()' or not. Useful
        e.g. for chatbots.
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=60.0,
        skip_prompt=True,
        # Originally, this was in
        # https://huggingface.co/spaces/Nymbo/Llama-3.2-1B-Instruct/blob/main/app.py
        # but it appears to not be used at all in streamers.py.
        #skip_special_tokens=True)
    )

    # In TextIteratorStream.
    assert streamer.timeout == 60.0
    # In TextStreamer,
    assert streamer.skip_prompt == True