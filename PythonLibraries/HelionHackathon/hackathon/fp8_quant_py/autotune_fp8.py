"""
autotune_fp8.py — Standalone autotuner for FP8 per-group quantization.

Run this on the Nebius B200 to find the best configs for each benchmark shape.
It will print a ready-to-paste SHAPE_CONFIGS block at the end.

Usage (on Nebius, with helion_env activated):
    cd ~/reference-kernels/problems/helion/fp8_quant_py
    python autotune_fp8.py                   # all benchmark shapes, default effort
    python autotune_fp8.py --effort quick    # faster (~3s/shape, fewer configs)
    python autotune_fp8.py --effort full     # thorough (~30s+/shape, recommended)

Output:
    - Prints best config per shape as each shape finishes (stderr from Helion)
    - At the end prints a complete SHAPE_CONFIGS dict ready to paste into
      fp8_group_quant_v2.py (or submission.py)
    - Also runs a quick correctness check after each shape

Workflow:
    1. Run this script on Nebius
    2. Copy the printed SHAPE_CONFIGS block into fp8_group_quant_v2.py
    3. Run:  python eval.py test fp8_quant_py/       # verify correctness
    4. Run:  python eval.py benchmark fp8_quant_py/  # measure performance
    5. Submit with popcorn
"""

import sys
import argparse
import torch
import helion
import helion.language as hl

# ── Shapes ────────────────────────────────────────────────────────────────────

# All 5 test shapes from task.yml (small, quick to autotune)
TEST_SHAPES = [
    # (num_tokens, hidden_dim, group_size, seed)
    (1,    256,   64,  4242),
    (4,    512,  128,  5236),
    (16,  1024,   64,  1001),
    (1,   4096,  128,  5531),
    (8,   4096,  128,  9173),
]

# 3 benchmark shapes from task.yml (these are what the leaderboard measures)
BENCHMARK_SHAPES = [
    (256,  4096, 128, 2146),
    (256,  8192, 128, 3129),
    (4096, 7168, 128, 54352),
]

FP8_MAX = 448.0
EPS     = 1e-10


# ── Kernel definition (NO config= → Helion autotuner takes over) ─────────────
# The autotune_effort= argument controls how thorough the search is.
# Helion's LFBO (Learning-based Function-space Bayesian Optimization) search
# explores: block_sizes, num_warps, num_stages, indexing (pointer / tensor_descriptor),
#           load_eviction_policies, reduction_loops, pid_type, loop_orders, l2_groupings.
# It prints the best config found to stderr as each shape completes.

def make_autotune_kernel(effort: str):
    """Return an autotunable kernel function with the given effort level.
    Valid effort values: 'none', 'quick', 'full'
    """
    @helion.kernel(static_shapes=True, autotune_effort=effort)
    def fp8_group_quant(
        data: torch.Tensor,        # [N, gsz]  — one row = one group
        scales_out: torch.Tensor,  # [N]       — written in-place
    ) -> torch.Tensor:
        # ── host code ──────────────────────────────────────────────────────
        nrows = data.size(0)
        # specialize: bake group_size as compile-time constant
        # → Helion generates warp-shuffle tree of EXACTLY log2(gsz) rounds
        # → no runtime masking, no loop overhead on the reduction
        ncols = hl.specialize(data.size(1))

        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

        # ── device code (compiles to one Triton kernel) ────────────────────
        for rr in hl.tile(nrows):
            # Load B0 rows, each of length gsz. Shape: [B0, gsz]
            row = data[rr, :].to(torch.float32)

            # ℓ∞ reduction: max|v_j| across the group (warp-shuffle tree)
            amax = torch.amax(torch.abs(row), -1)    # shape: [B0]
            amax = torch.clamp(amax, min=EPS)         # guard: no div-by-zero

            # scale σ = max(|v|, ε) / 448.0
            scale = amax / FP8_MAX                   # shape: [B0]

            # quantize: v / σ, clamp to FP8 range
            # scale[:, None] broadcasts [B0] → [B0, 1] for element-wise divide
            qout[rr, :] = torch.clamp(row / scale[:, None], -FP8_MAX, FP8_MAX)

            # write scales in-place (caller reads x_s directly)
            scales_out[rr] = scale

        return qout
    return fp8_group_quant


# ── Reference (pure PyTorch, for correctness check) ───────────────────────────
def reference(x: torch.Tensor, group_size: int):
    T, H = x.shape
    G    = H // group_size
    xg   = x.float().reshape(T, G, group_size)
    amax = xg.abs().amax(dim=-1).clamp(min=EPS)
    scale = amax / FP8_MAX
    q    = (xg / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(T, H), scale


# ── Main autotune loop ─────────────────────────────────────────────────────────
def run_autotune(effort: str, shapes: list, label: str):
    print(f"\n{'='*70}")
    print(f"  Autotuning {label} ({len(shapes)} shapes, effort='{effort}')")
    print(f"  Helion will print best config to stderr as each shape completes.")
    print(f"{'='*70}\n")

    kernel = make_autotune_kernel(effort)

    results = {}   # shape_key → (config_repr, time_ms)

    for T, H, G, seed in shapes:
        N   = T * (H // G)
        gsz = H // G
        shape_key = (T, H, G)

        print(f"  ▶  Shape T={T:>4}, H={H:>4}, gsz={gsz}, N={N:>7}  ...", flush=True)

        torch.manual_seed(seed)
        flat_in = torch.randn(N, gsz, device="cuda")
        flat_s  = torch.empty(N, device="cuda")

        # ── Trigger autotune: first call with a new (N, gsz) shape ────────
        # Helion's LFBO search runs here. Progress → stderr.
        # When it finishes, it prints:
        #   "One can hardcode the best config and skip autotuning with:"
        #   "  @helion.kernel(config=helion.Config(...))"
        flat_q = kernel(flat_in, flat_s)
        torch.cuda.synchronize()

        # ── Quick correctness check ────────────────────────────────────────
        # N = T * G where G = H // gsz (number of groups per token)
        # flat layout: [N, gsz] where N = T * G
        G_count = H // gsz   # groups per token (e.g. 4096/128 = 32)
        x_full  = torch.randn(T, H, device="cuda")
        ref_q, ref_s = reference(x_full, gsz)

        flat_x  = x_full.reshape(T * G_count, gsz)  # [N, gsz]
        flat_s2 = torch.empty(T * G_count, device="cuda")
        got_q_flat = kernel(flat_x, flat_s2)
        got_q = got_q_flat.reshape(T, H)
        got_s = flat_s2.reshape(T, G_count)

        q_ok = torch.allclose(got_q, ref_q, rtol=1e-3, atol=1e-3)
        s_ok = torch.allclose(got_s, ref_s, rtol=1e-3, atol=1e-3)
        status = "✅ PASS" if (q_ok and s_ok) else "❌ FAIL"
        print(f"     Correctness: {status}")

        results[shape_key] = "see stderr above"

    return results


# ── Pretty-print SHAPE_CONFIGS block ──────────────────────────────────────────
def print_config_template(shapes: list):
    """Print a template SHAPE_CONFIGS dict.
    After autotuning, paste the actual configs from stderr into the slots below.
    """
    print("\n" + "="*70)
    print("  PASTE YOUR AUTOTUNED CONFIGS HERE")
    print("  (copy from stderr output above — look for 'config=helion.Config(...)')")
    print("="*70)
    print()
    print("SHAPE_CONFIGS: dict[tuple, helion.Config] = {")
    for T, H, G, _ in shapes:
        print(f"    # T={T}, H={H}, gsz={G}  → paste config from stderr above")
        print(f"    ({T:>4}, {H:>4}, {G}): helion.Config(...),  # TODO")
    print("}")
    print()
    print("Then update fp8_group_quant_v2.py and run:")
    print("  python eval.py test      fp8_quant_py/   # must all PASS")
    print("  python eval.py benchmark fp8_quant_py/   # measure ms")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Autotune FP8 group quantization kernel on B200")
    parser.add_argument("--effort", default="quick",
                        choices=["none", "quick", "full"],
                        help="Autotune effort: quick≈3s/shape, full≈30s+/shape")
    parser.add_argument("--test-only", action="store_true",
                        help="Only autotune test shapes (skip benchmark shapes)")
    parser.add_argument("--bench-only", action="store_true",
                        help="Only autotune benchmark shapes")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. Run this on the Nebius B200 instance.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}  (sm_{cap[0]}{cap[1]})")

    shapes_to_run = []
    if not args.bench_only:
        shapes_to_run += [("Test shapes",    TEST_SHAPES)]
    if not args.test_only:
        shapes_to_run += [("Benchmark shapes", BENCHMARK_SHAPES)]

    for label, shapes in shapes_to_run:
        run_autotune(args.effort, shapes, label)

    # Print all shapes together for easy copy-paste
    all_shapes = ([] if args.bench_only  else TEST_SHAPES) + \
                 ([] if args.test_only   else BENCHMARK_SHAPES)
    print_config_template(all_shapes)


if __name__ == "__main__":
    main()
