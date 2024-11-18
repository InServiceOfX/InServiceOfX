from corecode.FileIO import get_project_directory_path
from moretransformers.Configurations import (
    GenerationConfiguration)
import yaml

class GroqGenerationConfiguration(GenerationConfiguration):
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "ThirdParties" / \
                "MoreLangchain" / "generation_configuration-groq.yml"
        ):
        super().__init__(configuration_path)

        if (self.timeout == ""):
            self.timeout = None
        # param max_tokens: int | None = None
        # Maximum number of tokens to generate.
        if (self.max_new_tokens == ""):
            self.max_new_tokens = None

        # See https://python.langchain.com/api_reference/groq/chat_models/langchain_groq.chat_models.ChatGroq.html
        # param temperature: float = 0.7
        # What sampling temperature to use.
        if (self.temperature == "" or self.temperature == None):
            self.temperature = 0.7

        with open(configuration_path, "r") as f:
            data = yaml.safe_load(f)
        # param max_tokens: int = 2
        # Maximum number of retries to make when generating.
        self.max_retries = data.get("max_retries", 2)