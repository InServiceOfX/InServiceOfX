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
    def _parse_dataset_generic(dataset: datasets.arrow_dataset.Dataset, 
                              filter_condition: callable, 
                              keys_to_include: list[str]):
        """
        Generic method to parse dataset with custom filter condition and keys.
        
        Args:
            dataset: The dataset to parse
            filter_condition: A function that takes a row and returns bool
            keys_to_include: List of string keys to include in the result
        """
        results = []

        for row in dataset:
            if filter_condition(row):
                result = {}
                for key in keys_to_include:
                    if key in row:
                        result[key] = row[key]
                results.append(result)
        return results

    @staticmethod
    def _parse_dataset(dataset: datasets.arrow_dataset.Dataset):
        def default_filter(row):
            return row["text"] is not None and row["text"] != ""
        
        keys_to_include = [
            "message_id", "parent_id", "lang", 
            "review_result", "role", "text"
        ]
        
        return ParseOpenAssistantOasst1._parse_dataset_generic(
            dataset, default_filter, keys_to_include)

    @staticmethod
    def _parse_for_prompter(dataset: datasets.arrow_dataset.Dataset):
        def prompter_filter(row):
            return (row["text"] is not None and row["text"] != "" and 
                   row["role"] == "prompter")
        
        keys_to_include = [
            "message_id", "parent_id", "lang", 
            "review_result", "text"
        ]
        
        return ParseOpenAssistantOasst1._parse_dataset_generic(
            dataset, prompter_filter, keys_to_include)

    @staticmethod
    def parse_for_train(dataset: datasets.DatasetDict):
        dataset_train = dataset["train"]
        return ParseOpenAssistantOasst1._parse_dataset(dataset_train)

    @staticmethod
    def parse_for_train_prompter(dataset: datasets.DatasetDict):
        dataset_train = dataset["train"]
        return ParseOpenAssistantOasst1._parse_for_prompter(dataset_train)

    @staticmethod
    def parse_for_train_prompter_english(dataset: datasets.DatasetDict):
        dataset_train = dataset["train"]
        def prompter_filter(row):
            return (
                row["text"] is not None and \
                    row["text"] != "" and \
                    row["role"] == "prompter" and \
                    row["lang"] == "en"
            )

        keys_to_include = [
            "message_id",
            "parent_id",
            "lang",
            "review_result",
            "text"
        ]
        return ParseOpenAssistantOasst1._parse_dataset_generic(
            dataset_train, prompter_filter, keys_to_include)

    @staticmethod
    def parse_for_validation(dataset: datasets.DatasetDict):
        dataset_validation = dataset["validation"]
        return ParseOpenAssistantOasst1._parse_dataset(dataset_validation)

    @staticmethod
    def parse_for_validation_prompter(dataset: datasets.DatasetDict):
        dataset_validation = dataset["validation"]
        return ParseOpenAssistantOasst1._parse_for_prompter(dataset_validation)

    @staticmethod
    def parse_for_validation_prompter_english(dataset: datasets.DatasetDict):
        dataset_validation = dataset["validation"]
        def prompter_filter(row):
            return (
                row["text"] is not None and \
                    row["text"] != "" and \
                    row["role"] == "prompter" and \
                    row["lang"] == "en"
            )

        keys_to_include = [
            "message_id",
            "parent_id",
            "lang",
            "review_result",
            "text"
        ]
        return ParseOpenAssistantOasst1._parse_dataset_generic(
            dataset_validation, prompter_filter, keys_to_include)