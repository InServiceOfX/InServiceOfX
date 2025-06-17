from corecode.Utilities import DataSubdirectories
from sentence_transformers import SentenceTransformer

from pathlib import Path

import numpy as np
import pytest
import torch

data_sub_dirs = DataSubdirectories()
MODEL_DIR = data_sub_dirs.Models / "Embeddings" / "BAAI" / \
    "bge-large-en-v1.5"
if not Path(MODEL_DIR).exists():
    print("for MODEL_DIR:", MODEL_DIR)
    print("MODEL_DIR.exists(): ", MODEL_DIR.exists())
    MODEL_DIR = Path("/Data1/Models/Embeddings/BAAI/bge-large-en-v1.5")

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

def test_SentenceTransformer_inits_with_path_as_string():
    """
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
    """

    # Must convert to string because otherwise
    # TypeError: argument of type 'PosixPath' is not iterable
    # /usr/local/lib/python3.12/dist-packages/sentence_transformers/SentenceTransformer.py:295: TypeError
    model = SentenceTransformer(
        str(MODEL_DIR),
        device = "cuda:0",
        # default is False
        trust_remote_code = True,
        )

    assert model.prompts == {}
    assert model.default_prompt_name == None
    assert model.similarity_fn_name == 'cosine'
    assert model.trust_remote_code == True
    assert model.truncate_dim == None
    for key, value in model.module_kwargs.items():
        print(key, value)

    for key, value in model._model_config.items():
        print(key, value)

    assert model.backend == "torch"

def test_sentence_transformers_usage():
    """
    https://www.sbert.net/
    """
    model = SentenceTransformer(str(MODEL_DIR), device = "cuda:0",)

    # The sentences to encode
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    # Calculate embeddings by calling model.encode()
    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L562
    # Defaults to True for convert_to_numpy.
    # convert_to_tensor Overwrites convert_to_numpy.
    embeddings = model.encode(sentences)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 1024)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    assert similarities.shape == (3, 3)
    assert isinstance(similarities, torch.Tensor)
    assert similarities.is_cuda == False
    assert similarities.is_cpu == True

    assert similarities[0, 0].item() == pytest.approx(0.9999998211860657)
    assert similarities[0, 1].item() == pytest.approx(
        0.8308399319648743,
        rel=1e-4)
    assert similarities[0, 2].item() == pytest.approx(
        0.477803498506546,
        rel=1e-4)
    assert similarities[1, 0].item() == pytest.approx(
        0.8308399319648743,
        rel=1e-4)
    assert similarities[1, 1].item() == pytest.approx(0.9999998211860657)
    assert similarities[1, 2].item() == pytest.approx(
        0.46056103706359863,
        rel=1e-4)
    assert similarities[2, 0].item() == pytest.approx(
        0.477803498506546,
        rel=1e-4)
    assert similarities[2, 1].item() == pytest.approx(
        0.46056103706359863,
        rel=1e-4)
    assert similarities[2, 2].item() == pytest.approx(0.9999998211860657)

    # On CUDA GPU
    embeddings = model.encode(sentences, convert_to_tensor = True)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, 1024)
    assert embeddings.is_cuda == True
    assert embeddings.is_cpu == False

    similarities = model.similarity(embeddings, embeddings)
    assert similarities.shape == (3, 3)
    assert isinstance(similarities, torch.Tensor)
    assert similarities.is_cuda == True
    assert similarities.is_cpu == False

def test_encode_with_normalize_embeddings():
    """
    https://huggingface.co/BAAI/bge-large-en-v1.5
    """
    sentences_1 = ["Sample Data-1", "Sample Data-2"]
    sentences_2 = ["Sample Data-3", "Sample Data-4"]

    model = SentenceTransformer(str(MODEL_DIR), device = "cuda:0",)

    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L562
    # normalize_embeddings, whether to normalize returned vectors to have length
    # 1. Defaults to False.

    embeddings_1 = model.encode(sentences_1, normalize_embeddings = True)
    embeddings_2 = model.encode(sentences_2, normalize_embeddings = True)

    assert embeddings_1.shape == (2, 1024)
    assert embeddings_2.shape == (2, 1024)

    similarities = embeddings_1 @ embeddings_2.T
    assert similarities.shape == (2, 2)

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

    similarities_2 = model.similarity(embeddings_1, embeddings_2)
    assert similarities_2.shape == (2, 2)
    assert similarities_2.is_cuda == False
    assert similarities_2.is_cpu == True

    assert similarities_2[0, 0].item() == pytest.approx(
        0.8752940893173218,
        rel=1e-4)
    assert similarities_2[0, 1].item() == pytest.approx(
        0.8796454668045044,
        rel=1e-5)
    assert similarities_2[1, 0].item() == pytest.approx(
        0.8713109493255615,
        rel=1e-5)
    assert similarities_2[1, 1].item() == pytest.approx(
        0.8681106567382812,
        rel=1e-4)
