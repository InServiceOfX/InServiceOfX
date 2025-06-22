from corecode.Utilities import DataSubdirectories

from embeddings.TextSplitters import get_max_token_limit
from pathlib import Path

data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

def test_get_max_token_limit():
    max_token_limit = get_max_token_limit(MODEL_DIR)
    assert max_token_limit == 512