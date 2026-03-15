"""
Causal Depthwise 1D Convolution — Helion Submission
Team: luminous-kernels

Algorithm (for each batch b, channel d, position t):
  out[b, d, t] = bias[d] + sum_{k=0}^{W-1} weight[d, k] * x[b, d, t - W + 1 + k]
  (out-of-bounds = 0 → handled by causal left-padding BEFORE kernel launch)

Kernel design:
  - Pre-pad input with W-1 zeros on the left (host side), so kernel has NO bounds checks
  - hl.tile([B, D, S], block_size=[1, None, None]):
      B dimension tiled at 1 (all benchmark shapes have B=1)
      D dimension (channels) tiled at block_sizes[0]
      S dimension (sequence) tiled at block_sizes[1]
  - hl.specialize(W): bake filter width into kernel → inner loop fully unrolled
  - Inner loop: for j in range(W): acc += weight[rd,j] * x_pad[bi, rd, rs+j]
      weight[rd,j] invariant over rs → compiler hoists load (weight reuse)
      x_pad[bi, rd, rs+j]: overlapping windows → compiler allocates shared memory
        halo of size S_tile + W - 1, loaded cooperatively once per block
  - Accumulate in f32, store back to output dtype
  - Zero inline_triton/asm (pure Helion DSL, 0% LOC escape hatch)

Configs: autotuned per shape on B200 Nebius. See SHAPE_CONFIGS below.
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

# ── Per-shape configs ──────────────────────────────────────────────────────
# Keys: (B, D, S, W)
# block_sizes=[D_tile, S_tile] for the two hl.tile None slots.
# Tuned for B200 (Nebius). Re-tune with eval.py --autotune on target GPU.

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # ── Test shapes ────────────────────────────────────────────────────────
    (1, 64,  64,  4): helion.Config(block_sizes=[16,  32],  num_warps=1, num_stages=1),
    (2, 128, 128, 4): helion.Config(block_sizes=[16,  64],  num_warps=2, num_stages=1),
    (1, 256, 256, 3): helion.Config(block_sizes=[16,  64],  num_warps=2, num_stages=1),
    (1, 128, 64,  8): helion.Config(block_sizes=[16,  32],  num_warps=2, num_stages=1),
    (4, 64,  128, 4): helion.Config(block_sizes=[16,  64],  num_warps=2, num_stages=1),

    # ── Benchmark shapes autotuned on Nebius B200 ─────────────────────────
    (1, 768,  512,  4): helion.Config(
        block_sizes=[16, 32],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[2],
        load_eviction_policies=['', '', ''],
        loop_orders=[[0, 1, 2]],
        num_stages=4,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 2],
        range_warp_specializes=[None, False],
        static_ranges=[False],
    ),
    (1, 768,  2048, 4): helion.Config(
        block_sizes=[32, 32],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['', '', ''],
        loop_orders=[[0, 1, 2]],
        num_stages=1,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None],
        static_ranges=[False],
    ),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[64, 128], num_warps=8, num_stages=3),
    (1, 2560, 2048, 4): helion.Config(
        block_sizes=[8, 128],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['first', 'last', ''],
        loop_orders=[[0, 1, 2]],
        num_stages=3,
        num_warps=1,
        pid_type='flat',
        range_flattens=[None, None],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None],
        static_ranges=[False],
    ),
    (1, 2560, 4096, 4): helion.Config(
        block_sizes=[8, 64],
        indexing=['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'],
        l2_groupings=[1],
        load_eviction_policies=['', '', ''],
        loop_orders=[[0, 2, 1]],
        num_stages=1,
        num_warps=1,
        pid_type='flat',
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None],
        static_ranges=[False],
    ),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[32, 128], num_warps=4, num_stages=2)


# ── Kernel factory ────────────────────────────────────────────────────────
_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def causal_conv1d(
        x_pad: torch.Tensor,   # [B, D, S+W-1] — zero-padded on left by W-1
        w: torch.Tensor,       # [D, W] — per-channel filter coefficients
        b: torch.Tensor,       # [D]   — per-channel bias
    ) -> torch.Tensor:
        # ── Host code ─────────────────────────────────────────────────────
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        # Specialize W: bake filter width as constant → fully unroll inner loop.
        # This eliminates loop overhead and lets compiler schedule W independent
        # load+fma chains (instruction-level parallelism).
        W = hl.specialize(w.size(1))
        S = L - W + 1                  # output sequence length

        y = torch.empty(B, D, S, dtype=x_pad.dtype, device=x_pad.device)

        # ── Device code ───────────────────────────────────────────────────
        # Tile over (B, D, S). block_size=[1, None, None] means:
        #   B: fixed size-1 tiles (scalar batch index)
        #   D: block_sizes[0] channels per block
        #   S: block_sizes[1] positions per block
        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin   # scalar batch index

            # f32 accumulator initialized to zero — one element per output position in tile
            acc = hl.zeros([rd, rs], dtype=torch.float32)

            # Inner loop over W taps — FULLY UNROLLED because W is specialized.
            # Generates W independent load+fma instruction sequences.
            for j in range(W):
                # Weight for tap j, shape [D_tile].
                # Invariant over rs → compiler hoists this load out of the S loop.
                # In practice: loaded once into registers, reused for all S_tile positions.
                coeff = w[rd, j].to(torch.float32)       # [D_tile]

                # Input at shifted position rs + j, shape [D_tile, S_tile].
                # Across all W taps, positions accessed: rs+0, rs+1, ..., rs+W-1
                # = S_tile positions + W-1 extra = halo of size S_tile+W-1.
                # Helion detects this overlap, allocates shared memory for the halo,
                # and cooperative-loads it once per block (no repeated global loads).
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)

                # Accumulate: weight broadcast over S axis, multiply elementwise
                acc = acc + x_val * coeff[:, None]       # [D_tile, S_tile]

            # Add per-channel bias, broadcast across sequence positions
            acc = acc + b[rd].to(torch.float32)[:, None]

            # Store tile output. acc is [D_tile, S_tile]; y[rb,rd,rs] is [1,D_tile,S_tile].
            # acc[None,:,:] adds the batch dim back.
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y
    return causal_conv1d

def _get_kernel(config: helion.Config):
    key = (tuple(config.block_sizes), config.num_warps, config.num_stages)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]


# ── Entry point ───────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    config = SHAPE_CONFIGS.get((B, D, S, W), FALLBACK_CONFIG)
    kernel = _get_kernel(config)

    # Causal left-padding: prepend W-1 zeros to sequence dimension.
    # Doing this on the host means the kernel has NO boundary checks at all.
    pad   = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    x_pad = torch.cat([pad, x], dim=2)   # [B, D, S + W - 1]

    return kernel(x_pad, weight, bias)
