"""
FP8 Per-Group Quantization — Helion Submission
Team: luminous-kernels

Algorithm:
  For each group of group_size contiguous elements per token row:
    1. absmax = max(|x_group|), clamped to eps to avoid div-by-zero
    2. scale = absmax / 448.0   (448.0 = FP8 E4M3 max)
    3. x_q   = clamp(x / scale, -448.0, 448.0)

Kernel design:
  - Reshape [T, H] → [N, gsz] so each row = exactly one group
  - hl.tile(nrows): one tile = one block of groups (parallelize over N)
  - hl.specialize(ncols): bake group_size into kernel as constant
    → reduction depth is known at compile time → warp-shuffle tree
  - torch.amax(torch.abs(row), -1): in-warp reduction, no extra hl.tile
  - scale[:, None]: broadcast [tile] → [tile, gsz] for vectorized divide
  - Zero inline_triton/asm (pure Helion DSL, 0% LOC escape hatch)

Configs: autotuned per shape on B200 Nebius. See SHAPE_CONFIGS below.
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

# ── Per-shape configs ──────────────────────────────────────────────────────
# Keys: (num_tokens, hidden_dim, group_size)
# Configs tuned for B200 (Nebius). To re-tune: run eval.py with --autotune
# on the target GPU, then paste the printed configs back here.

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # ── Test shapes (must pass correctness check) ─────────────────────────
    # Small: 1 token × 256 hidden, groups of 64 → N=4 groups total
    (1,    256,  64):  helion.Config(block_sizes=[4],  num_warps=1,  num_stages=1),
    # 4 tokens × 512 → N=16 groups
    (4,    512,  128): helion.Config(block_sizes=[8],  num_warps=2,  num_stages=1),
    # 16 tokens × 1024, gs=64 → N=256 groups
    (16,   1024, 64):  helion.Config(block_sizes=[16], num_warps=4,  num_stages=1),
    # 1 token × 4096 → N=32 groups
    (1,    4096, 128): helion.Config(block_sizes=[8],  num_warps=4,  num_stages=1),
    # 8 tokens × 4096 → N=256 groups
    (8,    4096, 128): helion.Config(block_sizes=[16], num_warps=4,  num_stages=2),

    # ── Benchmark shapes (performance scored) ────────────────────────────
    # Already covered above: (1, 4096, 128)
    # 16 tokens × 4096 → N=512 groups
    (16,   4096, 128): helion.Config(block_sizes=[32], num_warps=8,  num_stages=2),
    # 256 tokens × 4096 → N=8192 groups — saturates SM count on B200
    (256,  4096, 128): helion.Config(block_sizes=[64], num_warps=8,  num_stages=3),
    # 256 tokens × 8192 → N=16384 groups — wider hidden, same token count
    (256,  8192, 128): helion.Config(block_sizes=[64], num_warps=8,  num_stages=3),
    # 4096 tokens × 7168 → N=229376 groups — largest shape, max occupancy
    (4096, 7168, 128): helion.Config(block_sizes=[128],num_warps=8,  num_stages=4),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[32], num_warps=8, num_stages=2)


# ── Kernel factory ────────────────────────────────────────────────────────
_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fp8_group_quant(
        data: torch.Tensor,        # [N, gsz] — one row = one quantization group
        scales_out: torch.Tensor,  # [N]  — one scale per group (written in-place)
    ) -> torch.Tensor:
        # ── Host code (CPU, before kernel launch) ─────────────────────────
        nrows = data.size(0)
        # Specialize group_size: bake exact value into compiled kernel.
        # Enables: (a) warp-shuffle tree of known depth, (b) vectorized loads,
        # (c) no runtime masking on the inner reduction axis.
        ncols = hl.specialize(data.size(1))
        FP8_MAX = 448.0
        EPS     = 1e-10

        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

        # ── Device code (compiles to one Triton kernel) ────────────────────
        for rr in hl.tile(nrows):
            # Load tile of rows → shape [tile, gsz], cast to f32
            row = data[rr, :].to(torch.float32)

            # Per-group absmax: warp-shuffle reduction over the specialized ncols axis.
            # Helion sees ncols is a compile-time constant → depth-log2 shuffle tree.
            amax = torch.amax(torch.abs(row), -1)    # [tile]
            amax = torch.clamp(amax, min=EPS)         # guard zero-groups

            # Per-group scale factor
            scale = amax / FP8_MAX                   # [tile]

            # Quantize: broadcast scale, divide, clamp to FP8 range
            # scale[:, None] → [tile, 1] broadcasts over gsz columns
            qout[rr, :] = torch.clamp(row / scale[:, None], -FP8_MAX, FP8_MAX)

            # Write scale in-place (the caller reads x_s directly)
            scales_out[rr] = scale

        return qout
    return fp8_group_quant

def _get_kernel(config: helion.Config):
    key = (tuple(config.block_sizes), config.num_warps, config.num_stages)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]


# ── Entry point ───────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G    = x_s.shape[1]          # num groups per token
    gsz  = H // G                # group_size
    N    = T * G                 # total groups (kernel parallelism dimension)

    config = SHAPE_CONFIGS.get((T, H, gsz), FALLBACK_CONFIG)
    kernel = _get_kernel(config)

    flat_q = kernel(x.reshape(N, gsz), x_s.reshape(N))

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = x_s.reshape(T, G)   # already written in-place by kernel
    return x_q, x_s
