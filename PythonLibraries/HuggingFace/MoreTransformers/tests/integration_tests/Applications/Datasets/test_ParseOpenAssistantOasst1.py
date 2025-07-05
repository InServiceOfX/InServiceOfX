from corecode.Utilities import setup_datasets_path
from moretransformers.Applications.Datasets import ParseOpenAssistantOasst1
from moretransformers.Wrappers.Datasets import LoadAndSaveLocally

import datasets
import pytest

def path_for_OpenAssistant_oasst1():
    datasets_path = setup_datasets_path()
    return datasets_path / "OpenAssistant" / "oasst1"

@pytest.mark.skipif(
    not path_for_OpenAssistant_oasst1().exists(),
    reason="Dataset OpenAssistant/oasst1 not found"
)
def test_OpenAssistant_oasst1():
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    assert dataset is not None
    assert isinstance(dataset, datasets.DatasetDict)
    assert len(dataset) == 2
    assert "train" in dataset
    assert "validation" in dataset

    assert isinstance(dataset["train"], datasets.arrow_dataset.Dataset)
    assert len(dataset["train"]) == 84437
    assert len(dataset["validation"]) == 4401
    assert isinstance(dataset["train"][0], dict)
    keys = dataset["train"].column_names
    assert len(keys) == 18
    assert "message_id" in keys
    assert "parent_id" in keys
    assert "user_id" in keys
    assert "created_date" in keys
    assert "text" in keys
    assert "role" in keys
    assert "lang" in keys
    assert "review_count" in keys
    assert "review_result" in keys
    assert "deleted" in keys
    assert "rank" in keys
    assert "synthetic" in keys
    assert "model_name" in keys
    assert "detoxify" in keys
    assert "message_tree_id" in keys
    assert "tree_state" in keys
    assert "emojis" in keys
    assert "labels" in keys

    validation_keys = dataset["validation"].column_names
    assert len(validation_keys) == 18
    assert set(validation_keys) == set(keys)

    assert isinstance(dataset["train"][0]["message_id"], str)
    assert dataset["train"][0]["parent_id"] == None

    assert isinstance(dataset["train"][0]["user_id"], str)
    assert isinstance(dataset["train"][0]["created_date"], str)
    assert isinstance(dataset["train"][0]["text"], str)
    assert isinstance(dataset["train"][0]["role"], str)
    assert isinstance(dataset["train"][0]["lang"], str)
    assert isinstance(dataset["train"][0]["message_id"], str)
    assert isinstance(dataset["train"][0]["text"], str)
    assert isinstance(dataset["train"][0]["role"], str)
    assert isinstance(dataset["train"][0]["lang"], str)
    assert isinstance(dataset["train"][0]["review_count"], int)
    assert isinstance(dataset["train"][0]["review_result"], bool)
    assert isinstance(dataset["train"][0]["deleted"], bool)
    assert dataset["train"][0]["rank"] == None
    assert isinstance(dataset["train"][0]["synthetic"], bool)
    assert dataset["train"][0]["model_name"] == None
    assert isinstance(dataset["train"][0]["detoxify"], dict)

    train_roles = []
    for row in dataset["train"]:
        train_roles.append(row["role"])

    assert set(train_roles) == set(["prompter", "assistant"])

    validation_roles = []
    for row in dataset["validation"]:
        validation_roles.append(row["role"])

    assert set(validation_roles) == set(["prompter", "assistant"])

def test_ParseOpenAssistantOasst1_parse_for_train():
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    assert dataset is not None
    results = ParseOpenAssistantOasst1.parse_for_train(dataset)
    assert len(results) == 84437

def test_ParseOpenAssistantOasst1_parse_for_train_prompter():
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    assert dataset is not None
    results = ParseOpenAssistantOasst1.parse_for_train_prompter(dataset)
    assert len(results) == 31525

    # Uncomment to print the results
    # for i in range(10):
    #     print(
    #         results[i]["message_id"],
    #         results[i]["parent_id"],
    #         results[i]["text"],
    #     )