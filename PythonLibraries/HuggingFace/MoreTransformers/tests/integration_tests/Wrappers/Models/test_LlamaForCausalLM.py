from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

from safetensors import safe_open

from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaModel,
    LlamaConfig,
    LlamaPreTrainedModel)

from transformers.modeling_utils import (
    is_fsdp_enabled,
    is_safetensors_available,
    _add_variant,
    load_state_dict
)

from transformers.utils import SAFE_WEIGHTS_NAME

import pytest
import torch
import transformers

data_sub_dirs = DataSubdirectories()

SMOL_V2_MODEL_DIR = data_sub_dirs.ModelsLLM / "HuggingFaceTB" / \
    "SmolLM2-360M-Instruct"
SMOL_V2_skip_reason = f"Directory {SMOL_V2_MODEL_DIR} is empty or doesn't exist"

def test_LlamaForCausalLM_instantiates_without_quantization():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.model, LlamaModel)

    assert model.device == torch.device("cuda", index=0)

def test_LlamaForCausalLM_derives_from_LlamaPreTrainedModel():
    """
    See transformers, models/llama/modeling_llama.py

    Other than static class data members, this only implements
    def _init_weights(..).
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    assert model.config_class == LlamaConfig
    assert model.base_model_prefix == "model"
    assert model.supports_gradient_checkpointing == True
    assert model._no_split_modules == ["LlamaDecoderLayer"]
    assert model._skip_keys_device_placement == ["past_key_values"]
    assert model._supports_flash_attn_2 == True
    assert model._supports_sdpa == True
    assert model._supports_cache_class == True
    assert model._supports_quantized_cache == True

def test_LlamaForCausalLM_initializes_LlamaConfig():
    """
    See transformers, models/llama/configuration_llama.py for LlamaConfig,
    which derives from PretrainedConfig, found in
    transformers/configuration_utils.py
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    # Data members of PretrainedConfig

    assert model.config.bos_token_id == 128000
    assert model.config.pad_token_id == None
    assert model.config.eos_token_id == [128001, 128008, 128009]

def test_LlamaPreTrainedModel_fails_to_instantiate():
    """
    See transformers, src/transformers/modeling_utils.py
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    with pytest.raises(IndexError):
        model = LlamaPreTrainedModel.from_pretrained(pretrained_model_path)

def test_LlamaPreTrainedModel_derives_from_PreTrainedModel():
    """
    See transformers, src/transformers/modeling_utils.py
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    assert isinstance(model, LlamaPreTrainedModel)
    assert model.main_input_name == "input_ids"
    assert model.model_tags == None
    assert model._auto_class == None
    assert model._keep_in_fp32_modules == None
    assert model._keys_to_ignore_on_load_missing == None
    assert model._keys_to_ignore_on_load_unexpected == None
    assert model._keys_to_ignore_on_save == None
    # PreTrainedModel initiated this with None.
    assert model._tied_weights_keys == ['lm_head.weight']
    assert model.is_parallelizable == False
    # PretrainedModel initiated this with False.
    assert model.supports_gradient_checkpointing == True
    assert model._is_stateful == False
    # PretrainedModel initiated this with False.
    assert model._supports_static_cache == True
    # def is_fsdp_enabled() is implemented in src/transformers/modeling_utils.py
    # in def from_pretrained(..), this sets low_cpu_mem_usage=False.
    assert is_fsdp_enabled() == False
    # in def from_pretrained(..), this sets use_safetensors=True.
    assert is_safetensors_available() == True

def test_LlamaForCausalLM_from_pretrained_sets_config():
    """
    See transformers, src/transformers/modeling_utils.py
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    config = model.config_class.from_pretrained(
        pretrained_model_path)

    assert isinstance(config, LlamaConfig)
    assert getattr(config, "quantization_config", None) == None
    assert hasattr(config, "torch_dtype")

    assert config.name_or_path == ''

def test_LlamaForCausalLM_from_pretrained_instantiates():
    """
    See transformers, src/transformers/modeling_utils.py
    """
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)

    assert _add_variant(SAFE_WEIGHTS_NAME, None) == "model.safetensors"    

    # See def from_pretrained(..) in class PreTrainedModel of
    # src/transformers/modeling_utils.py
    resolved_archive_file = str(pretrained_model_path / "model.safetensors")

    f = safe_open(
        resolved_archive_file,
        framework="pt")
    
    metadata = f.metadata()
    assert metadata.get("format") == "pt"

    # in src/transformers/modeling_utils.py def load_state_dict(..) implemented
    # and returns torch.load(checkpoint_file, map_location=map_location, ..)

    state_dict = load_state_dict(resolved_archive_file)
    assert isinstance(state_dict, dict)
    expected_keys = [
        'model.embed_tokens.weight',
        'model.layers.0.input_layernorm.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.layers.0.mlp.up_proj.weight',
        'model.layers.0.post_attention_layernorm.weight',
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.self_attn.o_proj.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.v_proj.weight',
        'model.layers.1.input_layernorm.weight',
        'model.layers.1.mlp.down_proj.weight',
        'model.layers.1.mlp.gate_proj.weight',
        'model.layers.1.mlp.up_proj.weight',
        'model.layers.1.post_attention_layernorm.weight',
        'model.layers.1.self_attn.k_proj.weight',
        'model.layers.1.self_attn.o_proj.weight',
        'model.layers.1.self_attn.q_proj.weight',
        'model.layers.1.self_attn.v_proj.weight',
        'model.layers.10.input_layernorm.weight',
        'model.layers.10.mlp.down_proj.weight',
        'model.layers.10.mlp.gate_proj.weight',
        'model.layers.10.mlp.up_proj.weight',
        'model.layers.10.post_attention_layernorm.weight',
        'model.layers.10.self_attn.k_proj.weight',
        'model.layers.10.self_attn.o_proj.weight',
        'model.layers.10.self_attn.q_proj.weight',
        'model.layers.10.self_attn.v_proj.weight',
        'model.layers.11.input_layernorm.weight',
        'model.layers.11.mlp.down_proj.weight',
        'model.layers.11.mlp.gate_proj.weight',
        'model.layers.11.mlp.up_proj.weight',
        'model.layers.11.post_attention_layernorm.weight',
        'model.layers.11.self_attn.k_proj.weight',
        'model.layers.11.self_attn.o_proj.weight',
        'model.layers.11.self_attn.q_proj.weight',
        'model.layers.11.self_attn.v_proj.weight',
        'model.layers.12.input_layernorm.weight',
        'model.layers.12.mlp.down_proj.weight',
        'model.layers.12.mlp.gate_proj.weight', 'model.layers.12.mlp.up_proj.weight', 'model.layers.12.post_attention_layernorm.weight', 'model.layers.12.self_attn.k_proj.weight', 'model.layers.12.self_attn.o_proj.weight', 'model.layers.12.self_attn.q_proj.weight', 'model.layers.12.self_attn.v_proj.weight', 'model.layers.13.input_layernorm.weight', 'model.layers.13.mlp.down_proj.weight', 'model.layers.13.mlp.gate_proj.weight', 'model.layers.13.mlp.up_proj.weight', 'model.layers.13.post_attention_layernorm.weight', 'model.layers.13.self_attn.k_proj.weight', 'model.layers.13.self_attn.o_proj.weight', 'model.layers.13.self_attn.q_proj.weight', 'model.layers.13.self_attn.v_proj.weight', 'model.layers.14.input_layernorm.weight', 'model.layers.14.mlp.down_proj.weight', 'model.layers.14.mlp.gate_proj.weight', 'model.layers.14.mlp.up_proj.weight', 'model.layers.14.post_attention_layernorm.weight', 'model.layers.14.self_attn.k_proj.weight', 'model.layers.14.self_attn.o_proj.weight', 'model.layers.14.self_attn.q_proj.weight', 'model.layers.14.self_attn.v_proj.weight', 'model.layers.15.input_layernorm.weight', 'model.layers.15.mlp.down_proj.weight', 'model.layers.15.mlp.gate_proj.weight', 'model.layers.15.mlp.up_proj.weight', 'model.layers.15.post_attention_layernorm.weight', 'model.layers.15.self_attn.k_proj.weight', 'model.layers.15.self_attn.o_proj.weight', 'model.layers.15.self_attn.q_proj.weight', 'model.layers.15.self_attn.v_proj.weight', 'model.layers.2.input_layernorm.weight', 'model.layers.2.mlp.down_proj.weight', 'model.layers.2.mlp.gate_proj.weight', 'model.layers.2.mlp.up_proj.weight', 'model.layers.2.post_attention_layernorm.weight', 'model.layers.2.self_attn.k_proj.weight', 'model.layers.2.self_attn.o_proj.weight', 'model.layers.2.self_attn.q_proj.weight', 'model.layers.2.self_attn.v_proj.weight', 'model.layers.3.input_layernorm.weight', 'model.layers.3.mlp.down_proj.weight', 'model.layers.3.mlp.gate_proj.weight', 'model.layers.3.mlp.up_proj.weight', 'model.layers.3.post_attention_layernorm.weight', 'model.layers.3.self_attn.k_proj.weight', 'model.layers.3.self_attn.o_proj.weight', 'model.layers.3.self_attn.q_proj.weight', 'model.layers.3.self_attn.v_proj.weight', 'model.layers.4.input_layernorm.weight', 'model.layers.4.mlp.down_proj.weight', 'model.layers.4.mlp.gate_proj.weight', 'model.layers.4.mlp.up_proj.weight', 'model.layers.4.post_attention_layernorm.weight', 'model.layers.4.self_attn.k_proj.weight', 'model.layers.4.self_attn.o_proj.weight', 'model.layers.4.self_attn.q_proj.weight', 'model.layers.4.self_attn.v_proj.weight', 'model.layers.5.input_layernorm.weight', 'model.layers.5.mlp.down_proj.weight', 'model.layers.5.mlp.gate_proj.weight', 'model.layers.5.mlp.up_proj.weight', 'model.layers.5.post_attention_layernorm.weight', 'model.layers.5.self_attn.k_proj.weight', 'model.layers.5.self_attn.o_proj.weight', 'model.layers.5.self_attn.q_proj.weight', 'model.layers.5.self_attn.v_proj.weight', 'model.layers.6.input_layernorm.weight', 'model.layers.6.mlp.down_proj.weight', 'model.layers.6.mlp.gate_proj.weight', 'model.layers.6.mlp.up_proj.weight', 'model.layers.6.post_attention_layernorm.weight', 'model.layers.6.self_attn.k_proj.weight', 'model.layers.6.self_attn.o_proj.weight', 'model.layers.6.self_attn.q_proj.weight', 'model.layers.6.self_attn.v_proj.weight', 'model.layers.7.input_layernorm.weight', 'model.layers.7.mlp.down_proj.weight', 'model.layers.7.mlp.gate_proj.weight', 'model.layers.7.mlp.up_proj.weight', 'model.layers.7.post_attention_layernorm.weight', 'model.layers.7.self_attn.k_proj.weight', 'model.layers.7.self_attn.o_proj.weight', 'model.layers.7.self_attn.q_proj.weight', 'model.layers.7.self_attn.v_proj.weight', 'model.layers.8.input_layernorm.weight', 'model.layers.8.mlp.down_proj.weight', 'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.up_proj.weight', 'model.layers.8.post_attention_layernorm.weight', 'model.layers.8.self_attn.k_proj.weight', 'model.layers.8.self_attn.o_proj.weight', 'model.layers.8.self_attn.q_proj.weight', 'model.layers.8.self_attn.v_proj.weight', 'model.layers.9.input_layernorm.weight', 'model.layers.9.mlp.down_proj.weight', 'model.layers.9.mlp.gate_proj.weight', 'model.layers.9.mlp.up_proj.weight', 'model.layers.9.post_attention_layernorm.weight', 'model.layers.9.self_attn.k_proj.weight', 'model.layers.9.self_attn.o_proj.weight', 'model.layers.9.self_attn.q_proj.weight',
        'model.layers.9.self_attn.v_proj.weight',
        'model.norm.weight']
    
    for key in expected_keys:
        assert key in state_dict.keys()
    
    # upon __init__ with Contextmanagers,
    # model = cls(config, ...),
    # This state_dict appears to be used in cls._load_pretrained(..) when that
    # gets called.

@pytest.mark.skipif(
    is_directory_empty_or_missing(SMOL_V2_MODEL_DIR),
    reason=SMOL_V2_skip_reason
)
def test_use_LlamaForCausalLM_with_SmolLMv2():
    device = "cuda"
    tokenizer = GPT2TokenizerFast.from_pretrained(SMOL_V2_MODEL_DIR)
    model = LlamaForCausalLM.from_pretrained(
        SMOL_V2_MODEL_DIR,
        torch_dtype=torch.bfloat16).to(device)

    assert isinstance(model, LlamaForCausalLM)
    assert isinstance(model.model, LlamaModel)
    assert model.device == torch.device(device, index=0)

    messages = [
        {"role": "user", "content": "What is the capital of France."}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True).to(device)

    assert isinstance(model_inputs, transformers.tokenization_utils_base.BatchEncoding)

    # Without do_sample=True parameter value, then AttributeError occurs because
    # we specify top_p value.
    # TODO: Fix this,
    # RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasGemmStridedBatchedEx(handle, opa, opb, (int)m, (int)n, (int)k, (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea, b, CUDA_R_16BF, (int)ldb, strideb, (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec, (int)num_batches, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
    # outputs = model.generate(
    #     **model_inputs,
    #     max_new_tokens=50,
    #     temperature=0.2,
    #     top_p=0.6,
    #     do_sample=True)

    # assert isinstance(outputs, torch.Tensor)
    # assert outputs.shape == torch.Size([1, 45])
    # print(tokenizer.decode(outputs[0]))
