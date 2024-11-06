from transformers import AutoModelForCausalLM
from transformers.utils import is_peft_available

def create_AutoModelForCausalLM(
    model_subdirectory,
    torch_dtype=None,
    device_map="auto",
    quantization_config=None,
    trust_remote_code=True
    ):
    """
    Used in example code here:
    https://huggingface.co/spaces/Nymbo/Llama-3.2-1B-Instruct/blob/main/app.py

    Found directly mentioned both in utils/dummy_pt_objects.py and
    models/auto/modeling_auto.py. In latter, it says
    class AutoModelForCausalLM(_BaseAutoModelClass):
        _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    For the latter part, MODEL_FOR_CAUSAL_LM_MAPPING has key-value pair
    mentioning LlamaForCausalLM, and in config.json of huggingface repository
    for Llama-3.2-X it mentions "architectures": ["LlamaForCausalLM"].

    For _BaseAutoModelClass, it's in auto_factory.py, and has method
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    @param torch_dtype: See auto_factory.py, def from_pretrained(..),
    kwargs.get("torch_dtype", None), and kwargs.get("quantization_config", None)

    Example value: torch_dtype=torch.bfloat16

    @param trust_remote_code: see dynamic_module_utils.py, returns back
    trust_remote_code and only does something when it's None or False.
    """

    # This gets called in from_pretrained(..) of _BaseAutoModelClass. Use this
    # debug.
    is_peft_available_variable = is_peft_available()

    model = AutoModelForCausalLM.from_pretrained(
        model_subdirectory,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code)

    return model, is_peft_available_variable