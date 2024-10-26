from moretransformers.Configurations import Configuration

from colpali_engine.models import ColQwen2, ColQwen2Processor

from pathlib import Path
from PIL import Image

from transformers import BatchFeature
from transformers.models.qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor)

import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

configuration_colqwen2 = Configuration(
    test_data_directory / "configuration-colqwen2.yml")

# TODO: Allow this to run when enough GPU memory is available.
def test_ColQwen2_instantiates_with_cuda():

    assert Path(configuration_colqwen2.model_path).exists()

    # model = ColQwen2.from_pretrained(
    #     configuration_colqwen2.model_path,
    #     torch_dtype=configuration_colqwen2.torch_dtype,
    #     local_files_only=True,
    #     device_map="cuda:0").eval()

    # assert isinstance(model, ColQwen2)

def test_ColQwen2_instantiates_with_cpu_offloading():
    """
    From colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py,
    class ColQwen2(Qwen2VLForConditionalGeneration), and in
    transformers/models/qwen2_vl/modeling_qwen2_vl.py,
    class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel), and
    class Qwen2VLPreTrainedModel(PreTrainedModel). PreTrainedModel is in
    transformers/modeling_utils.py.

    def from_pretrained(
        cls,
        pretrained_model_name_or_path:
        *model_args,
        config:
        cache_dir:
        local_files_only,
        use_safetensors)
    """
    model = ColQwen2.from_pretrained(
        configuration_colqwen2.model_path,
        torch_dtype=configuration_colqwen2.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True).eval()

    assert isinstance(model, ColQwen2)
    assert isinstance(model, Qwen2VLForConditionalGeneration)
    assert model.dim == 128
    assert model.config.hidden_size == 1536
    assert model.padding_side == "left"
    assert model.patch_size == 14
    assert model.spatial_merge_size == 2
    assert model.vocab_size == 151936
    # TODO: This should be true but isn't.
    #assert model.get_input_embeddings() == nn.Embedding(151936, 1536)

def test_ColQwen2Processor_instantiates_with_cpu_offloading():
    processor = ColQwen2Processor.from_pretrained(
        configuration_colqwen2.model_path,
        local_files_only=True)

    assert isinstance(processor, ColQwen2Processor)
    assert isinstance(processor, Qwen2VLProcessor)
    assert processor.tokenizer.padding_side == "left"
    assert processor.min_pixels == 4 * 28 * 28
    assert processor.max_pixels == 768 * 28 * 28
    assert processor.factor == 28
    assert processor.max_ratio == 200

def test_ColQwen2Processor_processes_image():
    model = ColQwen2.from_pretrained(
        configuration_colqwen2.model_path,
        torch_dtype=configuration_colqwen2.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True,
        # Whether or not to offload the buffers with the model parameters. In
        # def from_pretrained(..)
        offload_buffers=True).eval()

    processor = ColQwen2Processor.from_pretrained(
        configuration_colqwen2.model_path,
        local_files_only=True)
    
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black")]

    # In colpali_engine/models/qwen2/colqwen2/processing_qwen2.py,
    # class ColQwen2Processor(Qwen2VLProcessor), method def process_images.
    batch_images = processor.process_images(images).to(model.device)

    assert isinstance(batch_images, BatchFeature)
    assert "input_ids" in batch_images
    assert "attention_mask" in batch_images
    assert "pixel_values" in batch_images
    assert "image_grid_thw" in batch_images

    assert batch_images.pixel_values.shape == torch.Size([2, 16, 1176])


def test_ColQwen2Processor_processes_queries():
    model = ColQwen2.from_pretrained(
        configuration_colqwen2.model_path,
        torch_dtype=configuration_colqwen2.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True,
        # Whether or not to offload the buffers with the model parameters. In
        # def from_pretrained(..)
        offload_buffers=True).eval()

    processor = ColQwen2Processor.from_pretrained(
        configuration_colqwen2.model_path,
        local_files_only=True)
    
    queries = [
        "Is attention really all you need?",
        "What is the amount of bananas farmed in Salvador?",]

    # In colpali_engine/models/qwen2/colqwen2/processing_qwen2.py,
    # class ColQwen2Processor(Qwen2VLProcessor), method def process_images.
    batch_queries = processor.process_queries(queries).to(model.device)

    assert isinstance(batch_queries, BatchFeature)
    assert "input_ids" in batch_queries
    assert "attention_mask" in batch_queries

    assert batch_queries.input_ids.shape == torch.Size([2, 33])


def test_ColQwen2Processor_processes_scores():
    model = ColQwen2.from_pretrained(
        configuration_colqwen2.model_path,
        torch_dtype=configuration_colqwen2.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True,
        offload_buffers=True).eval()

    processor = ColQwen2Processor.from_pretrained(
        configuration_colqwen2.model_path,
        local_files_only=True)

    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black")]

    queries = [
        "Is attention really all you need?",
        "What is the amount of bananas farmed in Salvador?",]

    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # Compute the MaxSim score (ColBERT-like) for given multi-vector query and
    # passage (image) embeddings.
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    assert scores.shape == torch.Size([2, 2])
    assert scores[0, 0].item() == 10.1875
    assert scores[0, 1].item() == 10.25
    assert scores[1, 0].item() == 9.8125
    assert scores[1, 1].item() == 9.875