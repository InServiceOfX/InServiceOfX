from corecode.Utilities import setup_datasets_path
from moretransformers.Wrappers.Datasets import LoadAndSaveLocally

import datasets
import pytest

def test_LoadAndSaveLocally_inits():
    load_and_save_locally = LoadAndSaveLocally()
    assert load_and_save_locally is not None
    assert load_and_save_locally._data == None
    assert load_and_save_locally._dataset_name == None
    assert load_and_save_locally._available_datasets == None

def test_LoadAndSaveLocally_get_available_datasets():
    load_and_save_locally = LoadAndSaveLocally()
    available_datasets = load_and_save_locally.get_available_datasets()
    assert available_datasets is not None

def test_LoadAndSaveLocally_loads_from_huggingface():
    load_and_save_locally = LoadAndSaveLocally()

    # https://huggingface.co/datasets/pisterlabs/promptset?library=datasets
    # ds = load_dataset("pisterlabs/promptset")
    dataset_name = "pisterlabs/promptset"
    dataset = load_and_save_locally.load_dataset(dataset_name)

    assert dataset is not None
    assert isinstance(dataset, datasets.DatasetDict)
    keys = dataset.keys()
    assert len(keys) == 1

    assert "train" in keys
    assert isinstance(dataset["train"], datasets.arrow_dataset.Dataset)
    assert len(dataset["train"]) == 93142
    assert isinstance(dataset["train"][0], dict)
    keys = dataset["train"].column_names
    assert len(keys) == 5
    assert "date_collected" in keys
    assert "repo_name" in keys
    assert "file_name" in keys
    assert "file_contents" in keys
    assert "prompts" in keys

    assert isinstance(dataset["train"][0]["prompts"], list)

    # https://huggingface.co/datasets/OpenAssistant/oasst1?library=datasets
    # ds = load_dataset("OpenAssistant/oasst1")
    dataset_name = "OpenAssistant/oasst1"
    dataset = load_and_save_locally.load_dataset(dataset_name)

    assert dataset is not None
    assert isinstance(dataset, datasets.DatasetDict)

def path_for_example_dataset_0():
    datasets_path = setup_datasets_path()
    return datasets_path / "pisterlabs" / "promptset"

@pytest.mark.skipif(
    not path_for_example_dataset_0().exists(),
    reason="Dataset pisterlabs/promptset not found"
)
def test_LoadAndSaveLocally_loads_from_disk():
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "pisterlabs/promptset"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    assert dataset is not None

@pytest.mark.skipif(
    not path_for_example_dataset_0().exists(),
    reason="Dataset pisterlabs/promptset not found"
)
def test_LoadAndSaveLocally_loads_from_disk_into_DatasetDict():
    datasets_path = setup_datasets_path()
    load_and_save_locally = LoadAndSaveLocally(datasets_path)
    dataset_name = "pisterlabs/promptset"
    dataset = load_and_save_locally.load_from_disk(dataset_name)
    assert dataset is not None

    assert isinstance(dataset, datasets.DatasetDict)
    assert len(dataset) == 1
    assert "train" in dataset
    assert isinstance(dataset["train"], datasets.arrow_dataset.Dataset)
    assert len(dataset["train"]) == 93142
    assert isinstance(dataset["train"][0], dict)
    keys = dataset["train"].column_names
    assert len(keys) == 5
    assert "date_collected" in keys
    assert "repo_name" in keys
    assert "file_name" in keys
    assert "file_contents" in keys
    assert "prompts" in keys

    assert isinstance(dataset["train"][0]["prompts"], list)
    assert len(dataset["train"][0]["prompts"]) == 0

    assert len(dataset["train"][1]["prompts"]) == 1
    assert isinstance(dataset["train"][1]["prompts"][0], str)

    assert len(dataset["train"][2]["prompts"]) == 4
    assert isinstance(dataset["train"][2]["prompts"][0], str)
    assert isinstance(dataset["train"][2]["prompts"][1], str)

    print(dataset["train"][2]["prompts"])

    assert len(dataset["train"][3]["prompts"]) == 0
