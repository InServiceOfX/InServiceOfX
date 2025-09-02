from pydantic import BaseModel

from typing import Any, Dict, Optional

class SamplingParameters(BaseModel):
    """
    See https://docs.sglang.ai/basic_usage/sampling_params.html
    """
    # https://docs.sglang.ai/basic_usage/sampling_params.html#core-parameters
    # Default 128 (!!!) Maximum output length measured in tokens.
    max_new_tokens: Optional[int] = None
    # https://platform.openai.com/docs/api-reference/chat/create#chat_create-temperature
    # Between 0 and 2. Higher values like 0.8 make output more random, while
    # lower values like 0.2 make it more focused and deterministic. OpenAI
    # recommends altering this or top_p, but not both.
    # SGLang says, temperature=0 corresponds to greedy sampling, a
    # higher termperature leads to more diversity.
    # SGLang defaults to 1.0.
    temperature: Optional[float] = None
    # https://docs.sglang.ai/basic_usage/sampling_params.html#core-parameters
    # Default 1.0. Top-p selects tokens from smallest sorted set whose
    # cumulative probability exceeds top_p.
    # When top_p = 1, this reduces to unrestricted smapling from all tokens.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    # OpenAI calls this nucleus sampling, where model considers results of the
    # tokens with top_p probability mass. So 0.1 means only tokens comprising
    # top 10% probability mass are considered.
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        dict_dump = self.model_dump()
        return {
            key: value for key, value in dict_dump.items()
            if value is not None}