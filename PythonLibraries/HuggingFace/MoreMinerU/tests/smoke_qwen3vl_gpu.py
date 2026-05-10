#!/usr/bin/env python3
"""GPU smoke test for ``Qwen3VLVLLM``.

Loads a Qwen3-VL checkpoint via the wrapper, runs one image+prompt through it,
prints the response, releases. Intended to be run *inside* the
``vllm-multimodal:25.06-py3`` container, with the InServiceOfX repo mounted at
``/InServiceOfX``.

Not pytest-collected (filename intentionally starts with ``smoke_``, not
``test_``). Run it directly:

    python /InServiceOfX/PythonLibraries/HuggingFace/MoreMinerU/tests/smoke_qwen3vl_gpu.py

Common variations:

    # baseline (4th-attempt config that OOM'd on a 12 GB 3060):
    python smoke_qwen3vl_gpu.py

    # AWQ-quantized weights (assumes you've downloaded them to /Data/...):
    python smoke_qwen3vl_gpu.py \\
        --model-path /Data/Models/Multimodal/Qwen/Qwen3-VL-4B-Instruct-AWQ \\
        --quantization awq

    # bigger GPU; raise context:
    python smoke_qwen3vl_gpu.py --max-model-len 8192 --max-num-seqs 4

    # smaller VLM to validate the wrapper logic with no memory pressure:
    python smoke_qwen3vl_gpu.py \\
        --model-path /Data/Models/Multimodal/Qwen/Qwen2.5-VL-3B-Instruct

The defaults match the 4th and most conservative attempt in the failure log
captured in MoreMinerU/README.md and the VLLMMultimodal STATUS.md decision
section. They will OOM on a 12 GB card with the un-quantized Qwen3-VL-4B; that
is the expected behavior — change the path or quantization to escape.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``import moremineru`` without installing the package.
THIS_FILE = Path(__file__).resolve()
LIBRARY_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(LIBRARY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-shot GPU smoke test for Qwen3VLVLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/Data/Models/Multimodal/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit"
        ),
        help=(
            "Directory containing the model's config.json + safetensors. "
            "Default is the AWQ-8bit community quant which fits on a 12 GB "
            "Ampere GPU; for un-quantized bf16 weights or a different quant, "
            "override + pass --quantization."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path(
            "/Workspace/Generated/CLIPDFExtraction/"
            "example-doc/"
            "page_1.png"
        ),
        help="Image to feed to the model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Describe what this engineering page shows. Identify the "
            "document title, revision, and owning organization."
        ),
        help="User-role text prompt.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Forwarded to vllm.LLM(dtype=...).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Forwarded to vllm.LLM(max_model_len=...).",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.90,
        help="Forwarded to vllm.LLM(gpu_memory_utilization=...).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Forwarded to vllm.LLM(max_num_seqs=...).",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help=(
            "Becomes limit_mm_per_prompt={'image': N} when N > 0. The "
            "default of 0 omits the limit entirely (vLLM picks its own)."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="compressed-tensors",
        help=(
            "Pass a vLLM-supported quantization tag. Default 'compressed-"
            "tensors' matches the cyankiwi AWQ-8bit model. For unquantized "
            "bf16 weights pass an empty string '' (or use a different "
            "model-path)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Sampling cap on the response length.",
    )
    parser.add_argument(
        "--no-eager",
        action="store_true",
        help=(
            "Disable enforce_eager — re-enables CUDA graphs. Faster after "
            "warmup, more VRAM. Default keeps eager mode for memory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from PIL import Image

    from moremineru.Configurations import Qwen3VLConfiguration
    from moremineru.Applications import Qwen3VLVLLM

    if not args.model_path.exists():
        print(
            f"error: model_path {args.model_path} does not exist (mount "
            f"the Data drive into the container?)",
            file=sys.stderr,
        )
        return 2
    if not args.image_path.exists():
        print(
            f"error: image_path {args.image_path} does not exist",
            file=sys.stderr,
        )
        return 2

    engine_kwargs = {
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "enforce_eager": not args.no_eager,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.limit_images > 0:
        engine_kwargs["limit_mm_per_prompt"] = {"image": args.limit_images}
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    config = Qwen3VLConfiguration(
        model_path=args.model_path,
        vllm_engine_kwargs=engine_kwargs,
        default_sampling_params={
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        },
    )

    print(">>> engine_kwargs:", engine_kwargs, flush=True)
    print(">>> sampling:", config.default_sampling_params, flush=True)

    model = Qwen3VLVLLM(config)
    print(">>> loading model...", flush=True)
    model.load()
    print(">>> loaded", flush=True)

    image = Image.open(args.image_path)
    print(">>> generating...", flush=True)
    response = model.generate(image, args.prompt)
    print("=" * 70)
    print(response)
    print("=" * 70)

    model.release()
    print(">>> released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
