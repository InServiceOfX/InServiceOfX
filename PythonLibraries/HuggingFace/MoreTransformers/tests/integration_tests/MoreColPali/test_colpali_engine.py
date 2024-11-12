from moretransformers.Configurations import Configuration

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset

from datasets import load_dataset

from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader

from transformers import BatchFeature

import torch

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

configuration_colpali = Configuration(
    test_data_directory / "configuration-colpali.yml")

# Originally this was 16 images.
# Only first 3 images used for demonstration purposes.
total_dataset = load_dataset("vidore/docvqa_test_subsampled", split="test[:3]")
images = total_dataset["image"]

query_indices = [1, 0]
queries = [total_dataset[idx]["query"] for idx in query_indices]

def test_ColPali_instantiates_with_cpu_offloading():
    assert Path(configuration_colpali.model_path).exists()

    model = ColPali.from_pretrained(
        configuration_colpali.model_path,
        torch_dtype=configuration_colpali.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True).eval()

    assert isinstance(model, ColPali)

    assert model.dim == 128
    assert model.config.hidden_size == 2048
    assert model.patch_size == 14

def test_ColPaliProcessor_instantiates_with_cpu_offloading():
    processor = ColPaliProcessor.from_pretrained(
        configuration_colpali.model_path,
        local_files_only=True)

    assert isinstance(processor, ColPaliProcessor)
    assert isinstance(processor, BaseVisualRetrieverProcessor)
    assert processor.tokenizer.padding_side == "right"


def test_ColPaliProcessor_processes_image_batches():
    processor = ColPaliProcessor.from_pretrained(
        configuration_colpali.model_path,
        local_files_only=True)

    assert isinstance(images, list)
    assert isinstance(images[0], Image.Image)

    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x))

    assert len(dataloader) == 3

    for batch in dataloader:
        assert isinstance(batch, BatchFeature)
        assert "pixel_values" in batch.keys()
        assert batch["pixel_values"].shape[1] == 3
        assert batch["pixel_values"].shape[2] == 448
        assert batch["pixel_values"].shape[3] == 448

def test_ColPaliProcessor_processes_queries():
    processor = ColPaliProcessor.from_pretrained(
        configuration_colpali.model_path,
        local_files_only=True)

    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x))

    assert len(dataloader) == 2

    for batch in dataloader:
        assert isinstance(batch, BatchFeature)
        assert "pixel_values" in batch.keys()

def test_ColPali_processes_image_batches():
    processor = ColPaliProcessor.from_pretrained(
        configuration_colpali.model_path,
        local_files_only=True)

    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x))

    model = ColPali.from_pretrained(
        configuration_colpali.model_path,
        torch_dtype=configuration_colpali.torch_dtype,
        device_map="auto",
        local_files_only=True,
        use_safetensors=True).eval()

    ds = []
    for batch_doc in dataloader:
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    assert len(ds) == len(images)
    assert ds[0].shape == (128,)
    assert ds[0].item() == pytest.approx(-0.001220703125, abs=1e-4)