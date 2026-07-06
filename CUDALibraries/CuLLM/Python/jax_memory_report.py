"""JAX-side memory report: XLA's own accounting of attention working sets.

The CUDA-side counterpart is AttentionMemoryReport (cudaMemGetInfo around
real cudaMallocs). This script asks the same question of the JAX
implementations in jax_attention_reference.py -- standard attention,
FlashAttention-1 (tiled, eager normalization), FlashAttention-2 (delayed
normalization), and the built-in op -- using two independent sources:

1. ``jax.jit(f).lower(...).compile().memory_analysis()`` -- XLA's *static*
   memory plan for the compiled computation: argument bytes, output bytes,
   and crucially TEMP bytes (the intermediates XLA decided to materialize,
   e.g. the B*H*N^2 score/weight buffers of standard attention). This is
   deterministic and per-computation -- no runtime noise.
2. Actually executing the compiled function, catching RESOURCE_EXHAUSTED
   live where the plan exceeds the card. The failure is a result.

Methodology gotcha this script handles for you: JAX preallocates 75% of
GPU memory by default, which makes every external measurement (nvidia-smi,
cudaMemGetInfo) useless. We set XLA_PYTHON_CLIENT_PREALLOCATE=false and
the ``platform`` allocator BEFORE importing jax, so allocations happen
on demand. This must run in the propulsion-with-cuda container:

    docker run --rm --gpus '"device=1"' -v <repo>:/InServiceOfX \
        propulsion-with-cuda:26.02-py3 bash
    cd /InServiceOfX && PYTHONPATH=CUDALibraries/CuLLM/Python \
        python3 CUDALibraries/CuLLM/Python/jax_memory_report.py
"""

from __future__ import annotations

import os

# Must be set before jax import; see module docstring.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# At N=4096 the standard/built-in paths' autotuner retries several GEMM
# configs against the OOM before giving up -- each retry logs a C++
# WARNING/ERROR block (bfc_allocator, config_assigner). That's expected,
# not a bug, but it drowns the one line that matters (the final
# "lowering/compile failed") in noise. Silence it here; the try/except in
# analyze() still reports the failure.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from functools import partial

import jax
import jax.numpy as jnp

from jax_attention_reference import (
    flash_attention,
    flash_attention2,
    jax_builtin_attention,
    make_cullm_inputs,
    standard_attention,
)

BATCH = 8
HEADS = 12
HEAD_DIM = 64
GIB = 1024.0 ** 3


def analyze(name: str, fn, q, k, v) -> None:
    """Print XLA's static memory plan for fn(q, k, v), then try running it."""
    try:
        compiled = jax.jit(fn).lower(q, k, v).compile()
    except Exception as error:  # noqa: BLE001 - report generation
        print(f"  {name:<34} lowering/compile failed: "
              f"{type(error).__name__}: {str(error)[:90]}")
        return

    analysis = compiled.memory_analysis()
    if analysis is None:
        print(f"  {name:<34} (memory_analysis unavailable on this backend)")
        return

    temp = analysis.temp_size_in_bytes
    arguments = analysis.argument_size_in_bytes
    output = analysis.output_size_in_bytes
    plan = (f"XLA plan: args {arguments / GIB:6.3f}  "
            f"out {output / GIB:6.3f}  TEMP {temp / GIB:7.3f} GiB")

    try:
        result = compiled(q, k, v)
        jax.block_until_ready(result)
        outcome = "ran OK"
        del result
    except Exception as error:  # noqa: BLE001 - the failure is the result
        message = str(error)
        outcome = ("execution FAILED: RESOURCE_EXHAUSTED (out of memory) "
                   "— the memory wall, measured"
                   if "RESOURCE_EXHAUSTED" in message
                   else f"execution failed: {message[:60]}")

    print(f"  {name:<34} {plan}  ->  {outcome}")


def main() -> None:
    device = jax.local_devices()[0]
    print(f"JAX {jax.__version__} | device: {device.device_kind} | "
          f"B*H = {BATCH * HEADS}, d = {HEAD_DIM} | "
          f"preallocate={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}, "
          f"allocator={os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']}")
    print()
    print("TEMP = intermediates XLA chose to materialize (the B*H*N^2 "
          "score/weight\nbuffers of standard attention live here; the "
          "tiled algorithms' temps are\nthe padded Q/K/V copies and "
          "per-tile loop state instead).")
    print()

    for n in (1024, 2048, 4096):
        q, k, v = make_cullm_inputs(BATCH, HEADS, n, HEAD_DIM)
        q16, k16, v16 = (x.astype(jnp.float16) for x in (q, k, v))
        print(f"N = {n} (fp32 args = "
              f"{3 * BATCH * HEADS * n * HEAD_DIM * 4 / GIB:.3f} GiB):")

        analyze(
            "standard attention (fp32)",
            partial(standard_attention, causal=False),
            q, k, v)
        analyze(
            "FlashAttention-1, lax tiles (fp32)",
            partial(flash_attention, block_size=64, causal=False),
            q, k, v)
        analyze(
            "FlashAttention-2, lax tiles (fp32)",
            partial(flash_attention2, block_size=64, causal=False),
            q, k, v)
        analyze(
            "built-in attention, xla (fp32)",
            partial(jax_builtin_attention, causal=False,
                    implementation="xla"),
            q, k, v)
        analyze(
            "built-in attention, cudnn (fp16)",
            partial(jax_builtin_attention, causal=False,
                    implementation="cudnn"),
            q16, k16, v16)
        print()

    print("CUDA-side counterpart (AttentionMemoryReport, measured via\n"
          "cudaMemGetInfo on the same RTX 3060): flash kernels (any rung) "
          "= Q,K,V,O\nonly, 0.375 GiB at N=4096; the standard baseline's "
          "two N^2 workspaces fail\nallocation live at 12.375 GiB "
          "requested. Compare against the TEMP column\nabove.")


if __name__ == "__main__":
    main()
