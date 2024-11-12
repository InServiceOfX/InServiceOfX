from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

from moretransformers.Wrappers.Models import run_model_generate
from moretransformers.Wrappers.Models.configure_tokens import get_pad_token_id

from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TextIteratorStreamer)

from typing import (List, Optional, Dict, Any)
import torch

class LocalLlama3(LLM):
    """
    From lib/core/langchain_core/language_models/llms.py, see class LLM(BaseLLM)
    where it says
    Simple interface for implementing a custom LLM.

    You should subclass this class and implement the following:
    - _call method: Run LLM on given prompt and input (used by invoke).
    - _identifying_params property: Return dictionary of identifying parameters.
    This is critical for caching and tracing.
    """
    tokenizer: PreTrainedTokenizerFast = Field(default=None, exclude=True)
    model: LlamaForCausalLM = Field(default=None, exclude=True)
    generation_configuration: GenerationConfiguration = Field(default=None, exclude=True)
    streamer: TextIteratorStreamer = Field(default=None, exclude=True)

    def __init__(
            self,
            configuration : Configuration,
            generation_configuration : GenerationConfiguration,
            device_map="auto",
            **kwargs):
        super().__init__(**kwargs)
        # Initialize tokenizer first
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            configuration.model_path)

        # See get_pad_token_id as a reference to these steps.
        pad_token_id = self.tokenizer.pad_token_id
        if (pad_token_id is None):
            pad_token_id = self.tokenizer.eos_token_id

        self.model = LlamaForCausalLM.from_pretrained(
            configuration.model_path,
            torch_dtype=configuration.torch_dtype,
            device_map=device_map,
            pad_token_id=pad_token_id)

        self.generation_configuration = generation_configuration

        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=generation_configuration.timeout,
            skip_prompt=True)

    def generate_for_llm_engine(
        self,
        messages : List[Dict[str, str]]):
        return_output = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True).to(self.model.device)

        with torch.no_grad():

            generate_output = run_model_generate(
                input_ids=return_output["input_ids"],
                model=self.model,
                streamer=self.streamer,
                eos_token_id=self.generation_configuration.eos_token_id,
                pad_token_id=get_pad_token_id(self.model, self.tokenizer),
                generation_configuration=self.generation_configuration,
                attention_mask=return_output["attention_mask"])

        output_buffer = ""
        for new_text in self.streamer:
            output_buffer += new_text

        # Strip the EOT token if present
        if self.tokenizer.eos_token in output_buffer:
            output_buffer = output_buffer.replace(self.tokenizer.eos_token, "").strip()

        return output_buffer

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at
                the first occurrence of any of the stop substrings.
            run_manager: Callback manager for run.
            **kwargs: Additional keyword arguments.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_for_llm_engine(messages)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Return a dictionary of identifying paramters. This is critical for
        caching and tracing purposes. Identifying parameters is a dict that
        identifies the LLM.
        """
        return {
            "model_name": self.model.config.name_or_path,
            "device_map": self.model.device_map,
            "torch_dtype": self.model.dtype
        }
    
    def _llm_type(self) -> str:
        return self.model.config.name_or_path