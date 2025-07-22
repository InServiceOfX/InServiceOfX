from pathlib import Path
from pydantic import BaseModel, Field
from typing import Union, Optional, Dict, Any, Literal

class FromPretrainedTokenizerConfiguration(BaseModel):
    """
    Pydantic BaseModel for configuring Hugging Face's from_pretrained() function
    for tokenizers.

    See src/transformers/tokenization_utils_base.py
    class PreTrainedTokenizerBase(..) and def from_pretrained(..)
        
    Required field: model_path
    Optional fields: All other parameters with sensible defaults
    """

    pretrained_model_name_or_path: Union[str, Path] = Field(
        ...,
        description=(
            "Typically this is a path of a directory containing vocabulary "
            "files required by the tokenizer."))

    force_download: Optional[bool] = Field(
        default=None,
        description=(
            "Whether or not to force the (re-)download the vocabulary files "
            "and override the cached versions if they exist."))

    local_files_only: bool = Field(
        default=True,
        description=(
            "Whether or not to only rely on local files and not to attempt to "
            "download any files"))
    
    trust_remote_code: bool = Field(
        default=True,
        description=(
            "Whether or not to allow for custom models defined on the Hub in "
            "their own modeling files. This option should only be set to "
            "`True` for repositories you trust and in which you have read the "
            "code, as it will execute code present on the Hub on your local "
            "machine."))

    def to_dict(self) -> Dict[str, Any]:
        config_dict = self.model_dump()
        return {k: v for k, v in config_dict.items() if v is not None}
