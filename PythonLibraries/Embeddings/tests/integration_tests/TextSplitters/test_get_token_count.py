from corecode.Utilities import DataSubdirectories, is_model_there

from embeddings.TextSplitters import get_token_count
from pathlib import Path

from transformers import AutoTokenizer

import json
import pytest

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

python_libraries_path = Path(__file__).parents[4]
test_data_path = python_libraries_path / "ThirdParties" / "APIs" / \
    "CommonAPI" / "tests" / "TestData"
test_conversation_path = test_data_path / "test_enable_thinking_true.json"

import sys
if str(test_data_path) not in sys.path:
    sys.path.append(str(test_data_path))

from CreateExampleConversation import CreateExampleConversation

def load_test_conversation():
    with open(test_conversation_path, "r") as f:
        return json.load(f)

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_get_token_count():
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
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

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_get_token_count_with_model_path():
    text = "This is a sample sentence to test tokenization."

    token_count = get_token_count(model_path, text)
    assert token_count == 10

    token_count = get_token_count(model_path, text, add_special_tokens=False)
    assert token_count == 10

    token_count = get_token_count(model_path, text, add_special_tokens=True)
    assert token_count == 12

    token_count = get_token_count(model_path, text, add_special_tokens=None)
    assert token_count == 10

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_get_token_count_of_conversation():

    conversation = CreateExampleConversation.EXAMPLE_CONVERSATION_0
    assert len(conversation) == 7

    assert get_token_count(model_path, conversation[0]["content"]) == 906
    assert get_token_count(model_path, conversation[1]["content"]) == 39
    assert get_token_count(model_path, conversation[2]["content"]) == 259
    assert get_token_count(model_path, conversation[3]["content"]) == 6
    assert get_token_count(model_path, conversation[4]["content"]) == 195
    assert get_token_count(model_path, conversation[5]["content"]) == 14
    assert get_token_count(model_path, conversation[6]["content"]) == 262

@pytest.mark.skipif(
    not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_get_token_count_of_long_text():
    conversation = load_test_conversation()
    assert len(conversation) == 16

    assert get_token_count(model_path, conversation[0]["content"]) == 45
    assert get_token_count(model_path, conversation[1]["content"]) == 1089
    assert get_token_count(model_path, conversation[2]["content"]) == 33
    assert get_token_count(model_path, conversation[3]["content"]) == 33
    assert get_token_count(model_path, conversation[4]["content"]) == 33
    assert get_token_count(model_path, conversation[5]["content"]) == 388
    assert get_token_count(model_path, conversation[6]["content"]) == 18
    assert get_token_count(model_path, conversation[7]["content"]) == 68
    assert get_token_count(model_path, conversation[8]["content"]) == 36
    assert get_token_count(model_path, conversation[9]["content"]) == 44
    assert get_token_count(model_path, conversation[10]["content"]) == 28
    assert get_token_count(model_path, conversation[11]["content"]) == 660
    assert get_token_count(model_path, conversation[12]["content"]) == 24
    assert get_token_count(model_path, conversation[13]["content"]) == 469
    assert get_token_count(model_path, conversation[14]["content"]) == 26
    assert get_token_count(model_path, conversation[15]["content"]) == 4
