from corecode.Utilities import DataSubdirectories

from embeddings.TextSplitters import get_token_count
from pathlib import Path

from transformers import AutoTokenizer

data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

def test_get_token_count():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    text = "This is a sample sentence to test tokenization."
    token_count = get_token_count(tokenizer, text)
    assert token_count == 10

    token_count = get_token_count(tokenizer, text, add_special_tokens=False)
    assert token_count == 10

    token_count = get_token_count(tokenizer, text, add_special_tokens=True)
    assert token_count == 12

    token_count = get_token_count(tokenizer, text, add_special_tokens=None)
    assert token_count == 10

def test_get_token_count_with_model_path():
    text = "This is a sample sentence to test tokenization."

    token_count = get_token_count(MODEL_DIR, text)
    assert token_count == 10

    token_count = get_token_count(MODEL_DIR, text, add_special_tokens=False)
    assert token_count == 10

    token_count = get_token_count(MODEL_DIR, text, add_special_tokens=True)
    assert token_count == 12

    token_count = get_token_count(MODEL_DIR, text, add_special_tokens=None)
    assert token_count == 10