"""Generate the CuLLM-vs-JAX attention benchmark report (markdown to stdout).

Run inside the propulsion-with-cuda container:

    cd /InServiceOfX/CUDALibraries/CuLLM/BuildDocker
    cmake ../Source && make WarpAttentionBenchmark AttentionReferenceDump -j4
    cd /InServiceOfX
    python3 CUDALibraries/CuLLM/Python/benchmark_report.py \
        --build-dir CUDALibraries/CuLLM/BuildDocker \
        > CUDALibraries/CuLLM/Documents/attention_benchmark_report_raw.md

What is measured — the single ATTENTION FORWARD CORE, O = softmax(QK^T/sqrt(d))V,
at multi-head scale (batch*heads = 96 independent slices, d = 64). This is NOT
full multi-head attention (no QKV/output projections), not backward, not GQA.

The matrix, per (sequence length, causal):
  float32 (same-precision, apples to apples):
    - CuLLM  warp-cooperative FlashAttention (hand-written CUDA, FA-2 math)
    - JAX/XLA fused standard attention (jit einsum+softmax; materializes N^2)
    - JAX built-in dot_product_attention, implementation="xla"
    - JAX FA-2 tiled reference (same math as CuLLM, expressed as lax loops)
  float16 (production-backend context):
    - CuLLM  warp-cooperative, T = __half (float accumulation)
    - cuDNN fused flash attention via jax.nn.dot_product_attention ("cudnn")

Timing: CUDA side uses cudaEvent around 20 launches after 3 warmups (see
warp_attention_benchmark.cu); JAX side uses wall clock around
block_until_ready after the same warmup count, which includes ~10-50 us of
dispatch overhead per call — negligible at n >= 1024, worth remembering when
reading the n = 256 row.
"""

from __future__ import annotations

import argparse
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_attention_reference import (
    benchmark,
    flash_attention2,
    jax_builtin_attention,
    make_cullm_inputs,
    max_abs_error,
    standard_attention,
)

BATCH = 8
HEADS = 12
HEAD_DIM = 64
SEQUENCE_LENGTHS = (256, 512, 1024, 2048, 4096)
# The XLA standard paths materialize (B, H, N, N) scores: 6.4 GB at n = 4096
# on a 12 GB card — skipped there, which is itself the IO-awareness argument.
MAX_STANDARD_N = 2048


def run_cullm(build_dir: Path) -> dict[tuple[str, int, int], float]:
    executable = build_dir / "WarpAttentionBenchmark"
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    timings: dict[tuple[str, int, int], float] = {}
    device_line = ""
    for line in result.stdout.splitlines():
        if line.startswith("Device:"):
            device_line = line
        if line.startswith("CSV "):
            dtype, n, batch_heads, causal, ms = line[4:].split(",")
            assert int(batch_heads) == BATCH * HEADS
            timings[(dtype, int(n), int(causal))] = float(ms)
    print(f"<!-- {device_line} -->")
    return timings


def timed(fn) -> float:
    try:
        return benchmark(fn)
    except Exception:  # noqa: BLE001 - report generation
        return float("nan")


def fmt(ms: float) -> str:
    return "—" if ms != ms else f"{ms:.3f}"


def report(build_dir: Path) -> None:
    print(f"<!-- JAX {jax.__version__}, devices {jax.devices()} -->")
    cullm = run_cullm(build_dir)

    for causal in (False, True):
        label = "causal" if causal else "non-causal"
        print()
        print(f"## Attention forward core, {label} "
              f"(B={BATCH}, H={HEADS}, d={HEAD_DIM}, float32, mean ms)")
        print()
        print("| N | CuLLM warp-coop CUDA | JAX/XLA standard (fused) | "
              "JAX built-in (xla) | JAX FA-2 tiled (lax loops) |")
        print("|---|---|---|---|---|")
        for n in SEQUENCE_LENGTHS:
            q, k, v = make_cullm_inputs(BATCH, HEADS, n, HEAD_DIM)
            if n <= MAX_STANDARD_N:
                standard_ms = timed(
                    partial(standard_attention, q, k, v, causal=causal))
                builtin_ms = timed(partial(
                    jax_builtin_attention, q, k, v, causal=causal,
                    implementation="xla"))
            else:
                standard_ms = float("nan")
                builtin_ms = float("nan")
            fa2_ms = timed(partial(
                flash_attention2, q, k, v, block_size=64, causal=causal))
            cullm_ms = cullm[("float32", n, int(causal))]
            print(f"| {n} | {fmt(cullm_ms)} | {fmt(standard_ms)} | "
                  f"{fmt(builtin_ms)} | {fmt(fa2_ms)} |")

    print()
    print(f"## float16 context (B={BATCH}, H={HEADS}, d={HEAD_DIM}, mean ms)")
    print()
    print("| N | causal | CuLLM warp-coop (fp16 I/O, fp32 accum) | "
          "cuDNN flash attention via JAX (fp16) |")
    print("|---|---|---|---|")
    for causal in (False, True):
        for n in SEQUENCE_LENGTHS:
            q, k, v = make_cullm_inputs(BATCH, HEADS, n, HEAD_DIM)
            q16, k16, v16 = (
                x.astype(jnp.float16) for x in (q, k, v))
            cudnn_ms = timed(partial(
                jax_builtin_attention, q16, k16, v16, causal=causal,
                implementation="cudnn"))
            cullm_ms = cullm[("float16", n, int(causal))]
            print(f"| {n} | {causal} | {fmt(cullm_ms)} | {fmt(cudnn_ms)} |")

    print()
    print("## Accuracy: CuLLM CUDA vs JAX FA-2, identical inputs (float32)")
    print()
    executable = build_dir / "AttentionReferenceDump"
    print("| N | d | causal | max abs difference |")
    print("|---|---|---|---|")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for n, head_dim, causal in (
            (64, 32, False), (100, 32, False), (128, 64, False),
            (150, 64, True),
        ):
            output_path = Path(tmp) / "dump.txt"
            subprocess.run(
                [str(executable), "--n", str(n), "--head-dim", str(head_dim),
                 "--causal", "1" if causal else "0",
                 "--output", str(output_path)],
                check=True)
            lines = output_path.read_text().splitlines()
            values = jnp.asarray(
                [float(v) for v in lines[1:]], dtype=jnp.float32)
            cullm_output = values.reshape(1, 1, n, head_dim)
            q, k, v = make_cullm_inputs(1, 1, n, head_dim)
            jax_output = flash_attention2(
                q, k, v, block_size=64, causal=causal)
            print(f"| {n} | {head_dim} | {causal} | "
                  f"{max_abs_error(cullm_output, jax_output):.3e} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir", type=Path,
        default=Path("CUDALibraries/CuLLM/BuildDocker"))
    args = parser.parse_args()
    report(args.build_dir)


if __name__ == "__main__":
    main()
