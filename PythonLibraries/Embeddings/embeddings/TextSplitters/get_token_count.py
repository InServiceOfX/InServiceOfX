from pathlib import Path
from transformers import AutoTokenizer
from typing import Optional

def get_token_count(
        model_tokenizer,
        text: str,
        add_special_tokens: Optional[bool] = None) -> int:

    if isinstance(model_tokenizer, str) or isinstance(model_tokenizer, Path):
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer,
            local_files_only = True)

    if add_special_tokens is None:
        return len(model_tokenizer.tokenize(text))
    return len(
        model_tokenizer.tokenize(text, add_special_tokens=add_special_tokens))

