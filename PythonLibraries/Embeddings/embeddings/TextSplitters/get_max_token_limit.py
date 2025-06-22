from transformers import AutoConfig

from pathlib import Path

def get_max_token_limit(model_path: str | Path) -> int:
    model_config = AutoConfig.from_pretrained(
        model_path,
        local_files_only = True,
        )
    return model_config.max_position_embeddings