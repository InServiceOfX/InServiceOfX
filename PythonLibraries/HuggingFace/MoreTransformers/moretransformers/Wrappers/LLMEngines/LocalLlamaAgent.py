from moretransformers.Configurations import Configuration

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

class LocalLlamaAgent:
    def __init__(
        self,
        configuration : Configuration,
        device_map="auto",
        trust_remote_code=True
        ):
        self.model = LlamaForCausalLM.from_pretrained(
            configuration.model_path,
            torch_dtype=configuration.torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code)
    
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            configuration.model_path)