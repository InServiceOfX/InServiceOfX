from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

@dataclass
class EmbeddingModelsConfiguration:
    """
    This dataclass allows for the (file) path for the embedding model(s) to be
    configurable. For instance,
    text_embedding_model is used in the init of TextSplitterByTokens and
    SentenceTransformer.
    """
    text_embedding_model: Union[str, Path]    

    @classmethod
    def from_yaml(cls, yaml_path: Path | str):
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(
            text_embedding_model=config["text_embedding_model"],
        )
