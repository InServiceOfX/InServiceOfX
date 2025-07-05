from moretransformers.Wrappers.Datasets import LoadAndSaveLocally
from pathlib import Path

def test_LoadAndSaveLocally_default_inits():
    load_and_save_locally = LoadAndSaveLocally()

    assert load_and_save_locally._save_path == str(Path.cwd())

def test_LoadAndSaveLocally__parse_dataset_name_works():
    load_and_save_locally = LoadAndSaveLocally()

    dataset_name = "pisterlabs/promptset"
    assert load_and_save_locally._parse_dataset_name(dataset_name) == \
        Path.cwd() / "pisterlabs" / "promptset"    
