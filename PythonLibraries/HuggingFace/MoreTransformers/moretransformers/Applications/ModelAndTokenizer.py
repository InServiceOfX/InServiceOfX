from pathlib import Path
from typing import Type, Optional
from moretransformers.Configurations import (
    FromPretrainedModelConfiguration,
    FromPretrainedTokenizerConfiguration,
    GenerationConfiguration)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

class ModelAndTokenizer:
    def __init__(
            self,
            model_path: str | Path,
            from_pretrained_model_configuration: \
                FromPretrainedModelConfiguration = None,
            from_pretrained_tokenizer_configuration: \
                FromPretrainedTokenizerConfiguration = None,
            generation_configuration: GenerationConfiguration = None,
            model_class: Type = None,
            tokenizer_class: Type = None):
        """
        Initialize ModelAndTokenizer with configurable model and tokenizer classes
        
        Args:
            model_path: Path to the model
            from_pretrained_model_configuration: Model configuration
            from_pretrained_tokenizer_configuration: Tokenizer configuration  
            model_class: Class to use for loading the model (defaults to AutoModel)
            tokenizer_class: Class to use for loading the tokenizer (defaults to AutoTokenizer)
        """
        self._model_path = model_path
        
        # Set default classes if not provided
        if model_class is None:
            self._model_class = AutoModelForCausalLM
        else:
            self._model_class = model_class
            
        if tokenizer_class is None:
            self._tokenizer_class = AutoTokenizer
        else:
            self._tokenizer_class = tokenizer_class

        # Set configurations
        if from_pretrained_model_configuration is None:
            self._fpmc = FromPretrainedModelConfiguration(
                pretrained_model_name_or_path=model_path)
        else:
            self._fpmc = from_pretrained_model_configuration

        if from_pretrained_tokenizer_configuration is None:
            self._fptc = FromPretrainedTokenizerConfiguration(
                pretrained_model_name_or_path=model_path)
        else:
            self._fptc = from_pretrained_tokenizer_configuration

        if generation_configuration is None:
            self._generation_configuration = \
                GenerationConfiguration(
                    max_new_tokens=100)
        else:
            self._generation_configuration = \
                generation_configuration

        self._model = None
        self._tokenizer = None

    def load_model(self, **kwargs):
        model_kwargs = self._fpmc.to_dict()
        model_kwargs.update(kwargs)
        self._model = self._model_class.from_pretrained(**model_kwargs)
    
    def load_tokenizer(self, **kwargs):
        tokenizer_kwargs = self._fptc.to_dict()
        tokenizer_kwargs.update(kwargs)
        self._tokenizer = self._tokenizer_class.from_pretrained(
            **tokenizer_kwargs)

    def apply_chat_template(
            self, 
            conversation,
            add_generation_prompt: Optional[bool] = None,
            tokenize: Optional[bool] = None,
            return_tensors: Optional[str] = None,
            to_device: Optional[bool] = None,
            **kwargs):
        """
        Args:
            add_generation_prompt: Typically True
            tokenize: Can be either False, resulting in a str, or True, resulting
            in a Dict-like object, with keys "input_ids" and "attention_mask".
            return_tensors: If None, defaults to 'pt'.

            kwargs:
            - return_dict: If True, returns a dict-like object, with keys
            "input_ids" and "attention_mask".
        """
        key_word_arguments = {}
        key_word_arguments.update(kwargs)
        key_word_arguments["conversation"] = conversation
        if add_generation_prompt is not None:
            key_word_arguments["add_generation_prompt"] = add_generation_prompt
        if tokenize is not None:
            key_word_arguments["tokenize"] = tokenize
        if return_tensors is not None:
            key_word_arguments["return_tensors"] = return_tensors
        else:
            key_word_arguments["return_tensors"] = 'pt'

        if to_device == False:
            return self._tokenizer.apply_chat_template(**key_word_arguments)

        if self._model is not None and self._model.device is not None:
            return self._tokenizer.apply_chat_template(
                **key_word_arguments).to(self._model.device)
        elif self._fpmc.device_map is not None:
            return self._tokenizer.apply_chat_template(
                **key_word_arguments).to(self._fpmc.device_map)
        else:
            return self._tokenizer.apply_chat_template(**key_word_arguments)

    def encode_by_calling_tokenizer(
            self,
            prompt_str,
            return_tensors: Optional[str] = None,
            padding: Optional[bool] = None,
            **kwargs):
        """
        From running integration tests, such as in test_Qwen_Qwen3-0_6B.py, this
        returns <class 'transformers.tokenization_utils_base.BatchEncoding'>,
        which has keys or attributes input_ids, attention_mask, device.
        """

        if return_tensors is None:
            return_tensors='pt'
        if padding is None:
            padding=True

        if self._model is not None and self._model.device is not None:
            return self._tokenizer(
                prompt_str,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs).to(self._model.device)
        elif self._fpmc.device_map is not None:
            return self._tokenizer(
                prompt_str,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs).to(self._fpmc.device_map)
        else:
            return self._tokenizer(
                prompt_str,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs)

    def move_encoded_to_device(self, encoded):
        if self._model is not None and self._model.device is not None:
            return {k: v.to(self._model.device) for k, v in encoded.items()}
        elif self._fpmc.device_map is not None:
            return {k: v.to(self._fpmc.device_map) for k, v in encoded.items()}
        else:
            return None

    def generate(self, input_ids, attention_mask=None, **kwargs):
        key_word_arguments = self._generation_configuration.to_dict()
        key_word_arguments.update(kwargs)
        key_word_arguments["input_ids"] = input_ids
        if attention_mask is not None:
            key_word_arguments["attention_mask"] = attention_mask

        return self._model.generate(**key_word_arguments)

    def decode_with_tokenizer(self, input, skip_special_tokens=None):
        if skip_special_tokens is None:
            skip_special_tokens = True

        return self._tokenizer.decode(
            input[0],
            skip_special_tokens=skip_special_tokens)

    def apply_chat_template_and_generate(
            self,
            conversation,
            with_attention_mask: Optional[bool] = None,
            skip_special_tokens: Optional[bool] = None,
            add_generation_prompt: Optional[bool] = None,
            return_tensors: Optional[str] = None
            ):
        if with_attention_mask is None:
            with_attention_mask = True

        if skip_special_tokens is None:
            skip_special_tokens = True

        if add_generation_prompt is None:
            add_generation_prompt = True

        if with_attention_mask:
            tokenizer_outputs = self.apply_chat_template(
                conversation,
                add_generation_prompt,
                tokenize=True,
                return_dict=True)

            output = self.generate(
                tokenizer_outputs["input_ids"],
                attention_mask=tokenizer_outputs["attention_mask"])

        else:
            tokenizer_outputs = self.apply_chat_template(
                conversation,
                add_generation_prompt,
                tokenize=True,
                return_tensors=return_tensors)

            output = self.generate(tokenizer_outputs)

        response = self.decode_with_tokenizer(
            output,
            skip_special_tokens=skip_special_tokens)

        return response

    def _parse_generate_output_into_thinking_and_content(
            self,
            model_inputs,
            generated_ids):
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        index = len(output_ids) - output_ids[::-1].index(151668)

        thinking_content = self._tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True)

        content = self._tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True)

        return (thinking_content, content)

    def generate_with_thinking_enabled(self, conversation):
        tokenizer_outputs = self.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=True)

        generated_ids = self.generate(
            tokenizer_outputs["input_ids"],
            attention_mask=tokenizer_outputs["attention_mask"])
        
        return self._parse_generate_output_into_thinking_and_content(
            tokenizer_outputs,
            generated_ids)