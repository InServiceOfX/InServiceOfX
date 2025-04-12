from moretransformers.Configurations import (
    Configuration,
    GenerationConfiguration)

from moretransformers.Wrappers.Models import run_model_generate
from moretransformers.Wrappers.Models.configure_tokens import get_pad_token_id

from typing import List, Dict

from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TextIteratorStreamer)

import torch

class LocalLlama:
    def __init__(
        self,
        configuration : Configuration,
        generation_configuration : GenerationConfiguration,
        device_map="auto"
        ):
        # Initialize tokenizer first
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            configuration.model_path)

        # Use the first token from eos_token_id list as pad_token_id
        pad_token_id = generation_configuration.eos_token_id[0] if isinstance(
            generation_configuration.eos_token_id, list) \
                else generation_configuration.eos_token_id

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
        messages : List[Dict[str, str]],
        ):
        #-> str:
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
            output_buffer = output_buffer.replace(
                self.tokenizer.eos_token, "").strip()

        return output_buffer

    def run(self, task, **kwargs):
        """Implement Agent interface"""
        messages = self._convert_task_to_messages(task)
        return self.generate_for_llm_engine(messages)

    def _convert_task_to_messages(self, task):
        """Convert task to chat messages format"""
        return [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": task}
        ]
