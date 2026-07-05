"""Generate the CuLLM-vs-JAX attention benchmark report (markdown to stdout).

Run inside the propulsion-with-cuda container:

    cd /InServiceOfX/CUDALibraries/CuLLM/BuildDocker
    cmake ../Source && make WarpAttentionBenchmark AttentionReferenceDump -j4
    cd /InServiceOfX
    python3 CUDALibraries/CuLLM/Python/benchmark_report.py \
        --build-dir CUDALibraries/CuLLM/BuildDocker \
        > CUDALibraries/CuLLM/Documents/attention_benchmark_report_raw.md

What is measured — the single ATTENTION FORWARD CORE,
O = softmax(QK^T/sqrt(d_head))V, at multi-head scale
(batch*heads = 96 independent slices, d_head = 64). This is NOT full
multi-head attention (no QKV/output projections), not backward, not GQA.

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


def fmt_ratio(numerator: float, denominator: float) -> str:
    if numerator != numerator or denominator != denominator or denominator == 0.0:
        return "—"
    return f"{numerator / denominator:.2f}x"


def gb_decimal(byte_count: int) -> float:
    return byte_count / 1_000_000_000.0


def memory_rows() -> None:
    print()
    print("## Memory scale (not runtime allocation accounting)")
    print()
    print("This table shows the algorithmic memory pressure that matters for "
          "the FlashAttention argument. It is deterministic from the tensor "
          "shapes; it is not `nvidia-smi` allocator reservation.")
    print()
    print("| N | standard score tensor `(B,H,N,N)` fp32 | "
          "Q/K/V/O tensors fp32 |")
    print("|---|---:|---:|")
    for n in SEQUENCE_LENGTHS:
        score_bytes = BATCH * HEADS * n * n * 4
        qkvo_bytes = 4 * BATCH * HEADS * n * HEAD_DIM * 4
        print(f"| {n} | {gb_decimal(score_bytes):.2f} GB | "
              f"{gb_decimal(qkvo_bytes):.2f} GB |")


def report(build_dir: Path) -> None:
    print(f"<!-- JAX {jax.__version__}, devices {jax.devices()} -->")
    cullm = run_cullm(build_dir)
    standard_timings: dict[tuple[int, bool], float] = {}
    builtin_xla_timings: dict[tuple[int, bool], float] = {}
    fa2_timings: dict[tuple[int, bool], float] = {}
    cudnn_timings: dict[tuple[int, bool], float] = {}
    accuracy_errors: list[float] = []

    print()
    print("## Benchmark shape")
    print()
    print(f"- `B={BATCH}` batch elements, `H={HEADS}` attention heads.")
    print(f"- `B*H={BATCH * HEADS}` independent attention slices run in parallel.")
    print(f"- `N` is context length / sequence length.")
    print(f"- `d_head={HEAD_DIM}` is the per-head Q/K/V dimension; "
          f"the implied model width for this benchmark is "
          f"`H*d_head={HEADS * HEAD_DIM}`.")
    print("- Timings are attention-core forward only: no QKV projection, "
          "no output projection, no backward pass.")
    memory_rows()

    for causal in (False, True):
        label = "causal" if causal else "non-causal"
        print()
        print(f"## Attention forward core, {label} "
              f"(B={BATCH}, H={HEADS}, d_head={HEAD_DIM}, "
              "float32, mean ms)")
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
            standard_timings[(n, causal)] = standard_ms
            builtin_xla_timings[(n, causal)] = builtin_ms
            fa2_timings[(n, causal)] = fa2_ms
            print(f"| {n} | {fmt(cullm_ms)} | {fmt(standard_ms)} | "
                  f"{fmt(builtin_ms)} | {fmt(fa2_ms)} |")

    print()
    print(f"## float16 context (B={BATCH}, H={HEADS}, "
          f"d_head={HEAD_DIM}, mean ms)")
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
            cudnn_timings[(n, causal)] = cudnn_ms
            print(f"| {n} | {causal} | {fmt(cullm_ms)} | {fmt(cudnn_ms)} |")

    print()
    print("## Accuracy: CuLLM CUDA vs JAX FA-2, identical inputs (float32)")
    print()
    executable = build_dir / "AttentionReferenceDump"
    print("| N | d_head | causal | max abs difference |")
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
            error = max_abs_error(cullm_output, jax_output)
            accuracy_errors.append(error)
            print(f"| {n} | {head_dim} | {causal} | {error:.3e} |")

    print()
    print("## Readout for presentation")
    print()
    n = 2048
    print("- `d_head=64` means each head uses 64-dimensional Q/K/V vectors; "
          "it is not the full model width.")
    print(f"- At `N={n}`, JAX/XLA standard attention is fast in float32 "
          f"({fmt(standard_timings[(n, False)])} ms) because XLA lowers the "
          "matmuls to optimized library kernels, but that path materializes "
          "the `B x H x N x N` score tensor.")
    print("- The `—` entries at `N=4096` are intentional: the standard paths "
          "would allocate about 6.4 GB just for fp32 scores "
          "`8*12*4096*4096*4`, before other tensors.")
    print(f"- CuLLM's causal tile skipping is visible at `N={n}`: "
          f"{fmt(cullm[('float32', n, 0)])} ms non-causal -> "
          f"{fmt(cullm[('float32', n, 1)])} ms causal "
          f"({fmt_ratio(cullm[('float32', n, 0)], cullm[('float32', n, 1)])}). "
          "JAX/XLA standard computes then masks, so its causal/non-causal "
          "times stay nearly equal.")
    print(f"- The fp16 cuDNN row is the production fused-flash baseline. At "
          f"`N={n}`, cuDNN is "
          f"{fmt_ratio(cullm[('float16', n, 0)], cudnn_timings[(n, False)])} "
          "faster than the scalar CuLLM fp16 path. Use "
          "`WarpAttentionBenchmark` for the WMMA and CuTe ladder that closes "
          "this gap.")
    if accuracy_errors:
        print(f"- Accuracy is a cross-language sanity check: max abs error is "
              f"at most `{max(accuracy_errors):.3e}` across the sampled "
              "CUDA-vs-JAX FA-2 cases.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir", type=Path,
        default=Path("CUDALibraries/CuLLM/BuildDocker"))
    args = parser.parse_args()
    report(args.build_dir)


if __name__ == "__main__":
    main()
