"""Compare CuLLM CUDA attention against JAX references.

Expected workflow inside the PropulsionWithCUDA container:

    cd /InServiceOfX/CUDALibraries/CuLLM/BuildGcc
    cmake ../Source
    make AttentionReferenceDump AttentionIOBenchmark -j4

    cd /InServiceOfX
    python3 CUDALibraries/CuLLM/Python/compare_cullm_jax_attention.py \
      --build-dir CUDALibraries/CuLLM/BuildGcc

The accuracy comparison is direct: CuLLM writes kernel output for deterministic
inputs, and JAX recomputes the same case from the same generator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_attention_reference import (
    benchmark,
    flash_attention,
    flash_attention2,
    jax_builtin_attention,
    make_cullm_inputs,
    max_abs_error,
    standard_attention,
)


def load_dump(path: Path) -> tuple[int, int, bool, jax.Array]:
    lines = path.read_text().splitlines()
    n_raw, head_dim_raw, causal_raw = lines[0].split()
    values = jnp.asarray([float(value) for value in lines[1:]], dtype=jnp.float32)
    n = int(n_raw)
    head_dim = int(head_dim_raw)
    causal = bool(int(causal_raw))
    return n, head_dim, causal, values.reshape(1, 1, n, head_dim)


def compare_accuracy(build_dir: Path) -> None:
    executable = build_dir / "AttentionReferenceDump"
    if not executable.exists():
        raise FileNotFoundError(
            f"{executable} not found. Build it with "
            "`make AttentionReferenceDump -j4` from the CuLLM build dir."
        )

    print("Accuracy: CuLLM warp-cooperative FlashAttention vs JAX FA-2")
    print(f"{'N':>5} {'d':>3} {'causal':>6} {'max |CuLLM-JAX|':>18}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for n, head_dim, causal in (
            (64, 32, False),
            (100, 32, False),
            (128, 64, False),
            (150, 64, True),
        ):
            output_path = tmp_path / f"cullm_n{n}_d{head_dim}_c{int(causal)}.txt"
            subprocess.run(
                [
                    str(executable),
                    "--n",
                    str(n),
                    "--head-dim",
                    str(head_dim),
                    "--causal",
                    "1" if causal else "0",
                    "--output",
                    str(output_path),
                ],
                check=True,
            )
            dump_n, dump_head_dim, dump_causal, cullm_output = load_dump(output_path)
            q, k, v = make_cullm_inputs(1, 1, dump_n, dump_head_dim)
            jax_output = flash_attention2(
                q,
                k,
                v,
                block_size=64,
                causal=dump_causal,
            )
            print(
                f"{dump_n:5d} {dump_head_dim:3d} {str(dump_causal):>6} "
                f"{max_abs_error(cullm_output, jax_output):18.3e}"
            )


def benchmark_jax() -> None:
    print()
    print("JAX performance references (mean ms)")
    print(
        f"{'B':>2} {'H':>2} {'N':>5} {'d':>3} {'causal':>6} "
        f"{'standard':>10} {'FA':>10} {'FA-2':>10} "
        f"{'builtin-xla':>12} {'builtin-cudnn':>13}"
    )
    for batch_size, heads, n, head_dim, causal in (
        (1, 1, 256, 64, False),
        (1, 1, 512, 64, False),
        (1, 1, 1024, 64, False),
        (1, 1, 1024, 64, True),
    ):
        q, k, v = make_cullm_inputs(batch_size, heads, n, head_dim)
        builtin_xla = jax.jit(
            lambda: jax_builtin_attention(
                q,
                k,
                v,
                causal=causal,
                implementation="xla",
            )
        )
        q16 = q.astype(jnp.float16)
        k16 = k.astype(jnp.float16)
        v16 = v.astype(jnp.float16)
        builtin_cudnn = jax.jit(
            lambda: jax_builtin_attention(
                q16,
                k16,
                v16,
                causal=causal,
                implementation="cudnn",
            )
        )

        def timed_or_nan(fn):
            try:
                return benchmark(fn)
            except Exception as error:  # noqa: BLE001 - benchmark reporting
                print(f"    skipped backend: {error}")
                return float("nan")

        standard_ms = timed_or_nan(
            lambda: standard_attention(q, k, v, causal=causal)
        )
        fa_ms = timed_or_nan(
            lambda: flash_attention(q, k, v, block_size=64, causal=causal)
        )
        fa2_ms = timed_or_nan(
            lambda: flash_attention2(q, k, v, block_size=64, causal=causal)
        )
        xla_ms = timed_or_nan(builtin_xla)
        # cuDNN flash attention in this JAX build requires fp16/bf16/fp8 inputs,
        # so this column is a half-precision backend reference, not an apples to
        # apples accuracy comparison with CuLLM's float32 benchmark.
        cudnn_ms = timed_or_nan(builtin_cudnn)
        print(
            f"{batch_size:2d} {heads:2d} {n:5d} {head_dim:3d} {str(causal):>6} "
            f"{standard_ms:10.3f} {fa_ms:10.3f} {fa2_ms:10.3f} "
            f"{xla_ms:12.3f} {cudnn_ms:13.3f}"
        )


def run_cullm_benchmark(build_dir: Path) -> None:
    executable = build_dir / "AttentionIOBenchmark"
    if not executable.exists():
        print()
        print(f"CuLLM performance benchmark skipped: {executable} not found.")
        return
    print()
    print("CuLLM performance benchmark")
    sys.stdout.flush()
    subprocess.run([str(executable)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("CUDALibraries/CuLLM/BuildGcc"),
        help="CuLLM CMake build directory containing AttentionReferenceDump.",
    )
    args = parser.parse_args()

    print(f"JAX {jax.__version__}")
    print(f"devices: {jax.devices()}")
    print()
    compare_accuracy(args.build_dir)
    benchmark_jax()
    run_cullm_benchmark(args.build_dir)


if __name__ == "__main__":
    main()
