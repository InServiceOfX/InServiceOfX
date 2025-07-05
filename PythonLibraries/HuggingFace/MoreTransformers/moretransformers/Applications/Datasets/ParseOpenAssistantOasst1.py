try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn(
        (
            "datasets is not installed. Please install it with `pip install `"
            "`datasets`"))

class ParseOpenAssistantOasst1:
    @staticmethod
    def _parse_dataset(dataset: datasets.arrow_dataset.Dataset):
        results = []

        for row in dataset:
            if row["text"] != None or row["text"] != "":
                result = {
                    "message_id": row["message_id"],
                    "parent_id": row["parent_id"],
                    "lang": row["lang"],
                    "review_result": row["review_result"],
                    "role": row["role"],
                    "text": row["text"],
                }
                results.append(result)
        return results

    @staticmethod
    def _parse_for_prompter(dataset: datasets.arrow_dataset.Dataset):
        results = []

        for row in dataset:
            if ((row["text"] != None or row["text"] != "") and \
                row["role"] == "prompter"):
                result = {
                    "message_id": row["message_id"],
                    "parent_id": row["parent_id"],
                    "lang": row["lang"],
                    "review_result": row["review_result"],
                    "text": row["text"],
                }
                results.append(result)
        return results

    @staticmethod
    def parse_for_train(dataset: datasets.DatasetDict):
        dataset_train = dataset["train"]
        return ParseOpenAssistantOasst1._parse_dataset(dataset_train)

    @staticmethod
    def parse_for_train_prompter(dataset: datasets.DatasetDict):
        dataset_train = dataset["train"]
        return ParseOpenAssistantOasst1._parse_for_prompter(dataset_train)

    @staticmethod
    def parse_for_validation(dataset: datasets.DatasetDict):
        dataset_validation = dataset["validation"]
        return ParseOpenAssistantOasst1._parse_dataset(dataset_validation)

    @staticmethod
    def parse_for_validation_prompter(dataset: datasets.DatasetDict):
        dataset_validation = dataset["validation"]
        return ParseOpenAssistantOasst1._parse_for_prompter(dataset_validation)