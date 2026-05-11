#!/usr/bin/env python3
"""GPU smoke test for ``ColQwen2_5Embedder``.

Loads the ColQwen2.5-v0.2 LoRA via colpali-engine, embeds a few P&ID page
images + a few text queries, runs MaxSim scoring, and prints the resulting
``[num_queries, num_images]`` score matrix. This proves the retrieval
primitive works end-to-end on the user's hardware.

Run *inside* the ``vllm-multimodal:25.06-py3`` container with the InServiceOfX
repo mounted at ``/InServiceOfX``:

    python /InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU/tests/smoke_colqwen_gpu.py

On first run, ``transformers`` will download the base model
(``vidore/colqwen2.5-base``, ~8 GB) into the HF cache if it isn't already
present. Subsequent runs are fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
LIBRARY_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(LIBRARY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU smoke test for ColQwen2_5Embedder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/Data/Models/Multimodal/vidore/colqwen2.5-v0.2"
        ),
        help="Path to the ColQwen2.5 LoRA adapter dir.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help=(
            "Image to embed. Repeat the flag to add multiple. Defaults to "
            "three rev11 P&ID pages from the example corpus."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        default=None,
        help=(
            "Query text. Repeat for multiple queries. Defaults to three "
            "P&ID-relevant queries."
        ),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda:0",
        help="device_map for from_pretrained.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="torch_dtype for from_pretrained.",
    )
    return parser.parse_args()


_DEFAULT_IMAGES = [
    Path(
        "/Workspace/Generated/CLIPDFExtraction/"
        "example-doc/"
        "page_1.png"
    ),
    Path(
        "/Workspace/Generated/CLIPDFExtraction/"
        "example-doc/"
        "page_2.png"
    ),
    Path(
        "/Workspace/Generated/CLIPDFExtraction/"
        "example-doc/"
        "page_3.png"
    ),
]

_DEFAULT_QUERIES = [
    "Title page or change log of an engineering document.",
    "Legend or symbol key for piping and instrumentation diagrams.",
    "Component list with valves, regulators, and tank specifications.",
]


def main() -> int:
    args = parse_args()

    from PIL import Image

    from moremineru.Configurations import ColQwen2_5Configuration
    from moremineru.Applications import ColQwen2_5Embedder

    if not args.model_path.exists():
        print(
            f"error: model_path {args.model_path} does not exist",
            file=sys.stderr,
        )
        return 2

    image_paths = args.image if args.image else _DEFAULT_IMAGES
    queries = args.query if args.query else _DEFAULT_QUERIES

    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(
            f"error: image(s) do not exist: {missing}",
            file=sys.stderr,
        )
        return 2

    print(f">>> images:  {len(image_paths)} ({image_paths[0].name}, ...)",
          flush=True)
    print(f">>> queries: {len(queries)}", flush=True)

    config = ColQwen2_5Configuration(
        model_path=args.model_path,
        torch_dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=None,
    )

    embedder = ColQwen2_5Embedder(config)
    print(">>> loading model (downloads base on first run, ~8 GB)...",
          flush=True)
    embedder.load()
    print(">>> loaded", flush=True)

    images = [Image.open(p) for p in image_paths]
    print(">>> embedding images...", flush=True)
    image_embeddings = embedder.embed_images(images)
    print(f">>> image_embeddings: {tuple(image_embeddings.shape)}",
          flush=True)

    print(">>> embedding queries...", flush=True)
    query_embeddings = embedder.embed_queries(queries)
    print(f">>> query_embeddings: {tuple(query_embeddings.shape)}",
          flush=True)

    print(">>> scoring...", flush=True)
    scores = embedder.score(query_embeddings, image_embeddings)
    print(f">>> scores shape: {tuple(scores.shape)}", flush=True)
    print("=" * 70)
    print(f"     {'  '.join(f'img{i+1:>3}' for i in range(len(image_paths)))}")
    for query_index, row in enumerate(scores):
        row_text = "  ".join(f"{value.item():6.2f}" for value in row)
        print(f"q{query_index+1}:  {row_text}   <- {queries[query_index][:60]}")
    print("=" * 70)

    embedder.release()
    print(">>> released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
