from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class GenerationConfiguration:
    """
    TODO: Consider using, from generation/configuration_utils.py,
    class GenerationConfig(PushToHubMixin) over this, depending upon the rules
    it takes to override default values.
    """
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
            "MoreTransformers" / "generation_configuration.yml"
    ):
        with open(str(configuration_path), 'r') as f:
            data = yaml.safe_load(f)

        self.timeout = data.get("timeout", 60.0)

        # Parameters that control length of output.
        # Max numbers of tokens to generate, ignoring number of tokens in the
        # prompt.
        self.max_new_tokens = data.get("max_new_tokens", 8192)

        # Parameters  for manipulation of the model output logits.

        # Value used to modulate next token probabilities.
        self.temperature = data.get("temperature", 1.0)

        # The number of highest probability vocabulary tokens to keep for
        # top-k-filtering.
        self.top_k = data.get("top_k", 20)
        if not isinstance(self.top_k, int):
            try:
                self.top_k = int(self.top_k)
            except ValueError:
                raise ValueError(f"top_k must be an integer, got {self.top_k}")

        # If set to float < 1, only smallest set of most probable tokens with
        # probabilities that add up to top_p  or higher are kept for generation.
        # Ensure top_p is a float
        self.top_p = data.get("top_p", 1.0)
        if not isinstance(self.top_p, float):
            try:
                self.top_p = float(self.top_p)
            except ValueError:
                raise ValueError(f"top_p must be a float, got {self.top_p}")

        # Special tokens that can be used at generation time.

        # ID of "end of sequence" token; optionally, use a list to set multiple
        # "end of sequence" tokens.
        # https://huggingface.co/spaces/Nymbo/Llama-3.2-1B-Instruct/blob/main/app.py
        # is where I obtained the default value.
        self.eos_token_id = data.get("eos_token_id", [1280001, 128008, 128009])
        if not isinstance(self.eos_token_id, list):
            raise ValueError(
                f"eos_token_id must be a list, got {type(self.eos_token_id)}")
