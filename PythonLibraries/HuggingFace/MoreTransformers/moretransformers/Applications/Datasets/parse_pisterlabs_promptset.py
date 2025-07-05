import warnings

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn(
        (
            "datasets is not installed. Please install it with `pip install `"
            "`datasets`"))

def parse_pisterlabs_promptset(dataset_dict: datasets.DatasetDict):
    dataset_train = dataset_dict["train"]

    results = []
    for row in dataset_train:
        row_repo_name = row["repo_name"]
        prompts = row["prompts"]
        for prompt in prompts:
            result = {
                "repo_name": row_repo_name,
                "prompt": prompt
            }
            results.append(result)

    return results