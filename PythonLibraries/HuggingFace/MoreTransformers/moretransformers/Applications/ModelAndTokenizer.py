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
            **kwargs):
        """
        Args:
            add_generation_prompt: Typically True
            tokenize: Can be either False, resuting in a str, or True, resulting
            in a Dict-like object, with keys "input_ids" and "attention_mask".
        """
        key_word_arguments = self._generation_configuration.to_dict()
        key_word_arguments.update(kwargs)
        key_word_arguments["conversation"] = conversation
        if add_generation_prompt is not None:
            key_word_arguments["add_generation_prompt"] = add_generation_prompt
        if tokenize is not None:
            key_word_arguments["tokenize"] = tokenize

        if self._model is not None and self._model.device is not None:
            return self._tokenizer.apply_chat_template(
                **key_word_arguments).to(self._model.device)
        else:
            return self._tokenizer.apply_chat_template(**key_word_arguments)

    def encode_by_calling_tokenizer(
            self,
            prompt_str,
            return_tensors: Optional[str] = None,
            padding: Optional[bool] = None,
            **kwargs):
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
        else:
            return self._tokenizer(
                prompt_str,
                return_tensors=return_tensors,
                padding=padding,
                **kwargs)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        key_word_arguments = self._generation_configuration.to_dict()
        key_word_arguments.update(kwargs)
        key_word_arguments["input_ids"] = input_ids
        if attention_mask is not None:
            key_word_arguments["attention_mask"] = attention_mask

        return self._model.generate(**key_word_arguments)
