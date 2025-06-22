from corecode.Utilities import DataSubdirectories

from pathlib import Path

import pytest
import torch
data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertTokenizerFast,
    )

from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    )

def test_tokenizer_inits():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    assert tokenizer is not None
    # https://github.com/huggingface/transformers/blob/e61160c5dbd470c4644e6c248d2acb64f763b6d5/src/transformers/models/bert/tokenization_bert_fast.py#L32
    assert isinstance(tokenizer, BertTokenizerFast)
    assert tokenizer.do_lower_case == True

def test_model_inits():
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    # https://github.com/huggingface/transformers/blob/e61160c5dbd470c4644e6c248d2acb64f763b6d5/src/transformers/models/bert/modeling_bert.py#L864
    assert isinstance(model, BertModel)

    # BertModel __init__
    print(model.config)
    assert "_attn_implementation_autoset" in model.config
    assert "architectures" in model.config
    assert isinstance(model.embeddings, BertEmbeddings)
    assert isinstance(model.encoder, BertEncoder)
    assert model.pooler is not None
    assert isinstance(model.pooler, BertPooler)
    assert model.attn_implementation == "sdpa"
    assert model.position_embedding_type == "absolute"

    model.to("cuda:0")
    assert model.device.type == "cuda"

def test_tokenizer_and_model_usage():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )

    sentences_1 = ["Sample Data-1", "Sample Data-2"]
    sentences_2 = ["Sample Data-3", "Sample Data-4"]

    # Tokenize sentences
    encoded_input_1 = tokenizer(
        sentences_1,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )
    encoded_input_2 = tokenizer(
        sentences_2,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )

    assert "input_ids" in encoded_input_1
    assert "attention_mask" in encoded_input_1

    # for s2p(short query to long passage) retrieval task, add an instruction to
    # query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries],
    # padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        model_output_1 = model(**encoded_input_1)
        model_output_2 = model(**encoded_input_2)

        # Perform pooling. In this case, cls pooling.
        sentence_embeddings_1 = model_output_1[0][:, 0, :]
        sentence_embeddings_2 = model_output_2[0][:, 0, :]

    # normalize embeddings
    sentence_embeddings_1 = torch.nn.functional.normalize(
        sentence_embeddings_1,
        p = 2,
        dim = 1,
        )
    sentence_embeddings_2 = torch.nn.functional.normalize(
        sentence_embeddings_2,
        p = 2,
        dim = 1,
        )

    assert isinstance(sentence_embeddings_1, torch.Tensor)
    assert sentence_embeddings_1.shape == (2, 1024)
    assert sentence_embeddings_1.is_cuda == False
    assert sentence_embeddings_1.is_cpu == True

def test_tokenizer_and_model_usage_on_gpu():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model.to("cuda:0")

    sentences_1 = ["Sample Data-1", "Sample Data-2"]
    sentences_2 = ["Sample Data-3", "Sample Data-4"]

    # Tokenize sentences
    encoded_input_1 = tokenizer(
        sentences_1,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )
    encoded_input_2 = tokenizer(
        sentences_2,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )

    # Move to GPU
    encoded_input_1 = {k: v.to("cuda:0") for k, v in encoded_input_1.items()}
    encoded_input_2 = {k: v.to("cuda:0") for k, v in encoded_input_2.items()}

    assert "input_ids" in encoded_input_1
    assert "attention_mask" in encoded_input_1

    # Compute token embeddings
    with torch.no_grad():
        # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        model_output_1 = model(**encoded_input_1)
        model_output_2 = model(**encoded_input_2)

        # Perform pooling. In this case, cls pooling.
        sentence_embeddings_1 = model_output_1[0][:, 0, :]
        sentence_embeddings_2 = model_output_2[0][:, 0, :]

    # normalize embeddings
    sentence_embeddings_1 = torch.nn.functional.normalize(
        sentence_embeddings_1,
        p = 2,
        dim = 1,
        )
    sentence_embeddings_2 = torch.nn.functional.normalize(
        sentence_embeddings_2,
        p = 2,
        dim = 1,
        )

    assert isinstance(sentence_embeddings_1, torch.Tensor)
    assert sentence_embeddings_1.shape == (2, 1024)
    assert sentence_embeddings_1.is_cuda == True
    assert sentence_embeddings_1.is_cpu == False

def test_tokenizer_and_model_output_computes_similarities():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )

    sentences_1 = ["Sample Data-1", "Sample Data-2"]
    sentences_2 = ["Sample Data-3", "Sample Data-4"]

    # Tokenize sentences
    encoded_input_1 = tokenizer(
        sentences_1,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )
    encoded_input_2 = tokenizer(
        sentences_2,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )

    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
        model_output_2 = model(**encoded_input_2)

        sentence_embeddings_1 = model_output_1[0][:, 0, :]
        sentence_embeddings_2 = model_output_2[0][:, 0, :]

    # normalize embeddings
    sentence_embeddings_1 = torch.nn.functional.normalize(
        sentence_embeddings_1,
        p = 2,
        dim = 1,
        )
    sentence_embeddings_2 = torch.nn.functional.normalize(
        sentence_embeddings_2,
        p = 2,
        dim = 1,
        )

    similarities = sentence_embeddings_1 @ sentence_embeddings_2.T
    assert similarities.shape == (2, 2)
    assert similarities.is_cuda == False
    assert similarities.is_cpu == True

    assert similarities[0, 0].item() == pytest.approx(
        0.8752940893173218,
        rel=1e-4)
    assert similarities[0, 1].item() == pytest.approx(
        0.8796454668045044,
        rel=1e-5)
    assert similarities[1, 0].item() == pytest.approx(
        0.8713109493255615,
        rel=1e-5)
    assert similarities[1, 1].item() == pytest.approx(
        0.8681105375289917,
        rel=1e-4)

def test_tokenizer_and_model_output_computes_similarities_on_gpu():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )
    model.to("cuda:0")

    sentences_1 = ["Sample Data-1", "Sample Data-2"]
    sentences_2 = ["Sample Data-3", "Sample Data-4"]

    # Tokenize sentences
    encoded_input_1 = tokenizer(
        sentences_1,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )
    encoded_input_2 = tokenizer(
        sentences_2,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        )

    # Move to GPU
    encoded_input_1 = {k: v.to("cuda:0") for k, v in encoded_input_1.items()}
    encoded_input_2 = {k: v.to("cuda:0") for k, v in encoded_input_2.items()}

    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
        model_output_2 = model(**encoded_input_2)

        sentence_embeddings_1 = model_output_1[0][:, 0, :]
        sentence_embeddings_2 = model_output_2[0][:, 0, :]

    # normalize embeddings
    sentence_embeddings_1 = torch.nn.functional.normalize(
        sentence_embeddings_1,
        p = 2,
        dim = 1,
        )
    sentence_embeddings_2 = torch.nn.functional.normalize(
        sentence_embeddings_2,
        p = 2,
        dim = 1,
        )

    similarities = sentence_embeddings_1 @ sentence_embeddings_2.T
    assert similarities.shape == (2, 2)
    assert similarities.is_cuda == True
    assert similarities.is_cpu == False

    # Move values to CPU so to compare values on the host CPU.
    similarities_cpu = similarities.cpu()
    assert similarities_cpu.shape == (2, 2)
    assert similarities_cpu.is_cuda == False
    assert similarities_cpu.is_cpu == True

    assert similarities_cpu[0, 0].item() == pytest.approx(
        0.8752940893173218,
        rel=1e-4)
    assert similarities_cpu[0, 1].item() == pytest.approx(
        0.8796454668045044,
        rel=1e-4)
    assert similarities_cpu[1, 0].item() == pytest.approx(
        0.8713109493255615,
        rel=1e-4)
    assert similarities_cpu[1, 1].item() == pytest.approx(
        0.8681105375289917,
        rel=1e-4)

def test_get_token_count_steps():
    """Show explicitly the steps of get_token_count to aid development."""
    model = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )

    text = "This is a sample sentence to test tokenization."

    token_count = len(model.tokenize(text, add_special_tokens=False))
    assert token_count == 10

    token_count = len(model.tokenize(text, add_special_tokens=True))
    assert token_count == 12

    token_count = len(model.tokenize(text))
    assert token_count == 10

    character_count = len(text)
    assert character_count == 47

    print(f"Characters per token: {character_count / token_count:.2f}")

    model = BertTokenizerFast.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )

    token_count = len(model.tokenize(text, add_special_tokens=False))
    assert token_count == 10

    token_count = len(model.tokenize(text, add_special_tokens=True))
    assert token_count == 12

    token_count = len(model.tokenize(text))
    assert token_count == 10

    print(f"Characters per token: {character_count / token_count:.2f}")

def test_get_max_token_limit_steps():
    """Show explicitly the steps of get_max_token_limit to aid development."""
    model_config = AutoConfig.from_pretrained(
        MODEL_DIR,
        local_files_only = True,
        )

    max_token_limit = model_config.max_position_embeddings
    assert max_token_limit == 512