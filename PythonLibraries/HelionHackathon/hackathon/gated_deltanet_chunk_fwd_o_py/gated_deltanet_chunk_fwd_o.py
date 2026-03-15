#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius
"""
Gated DeltaNet chunk_fwd_o — Helion Submission
Team: luminous-kernels

Algorithm (output computation):
  For each chunk c independently:
    inter = (q @ h[c]) * exp(g)                    — state contribution
    intra = causal_mask(q @ k^T * exp(g_i - g_j)) @ v_new  — local attention
    output = (inter + intra) * scale               — combined, scale = K^{-0.5}

Kernel design:
  - hl.tile([B*H, T], block_size=[1, C=64]): each tile = one chunk
  - Two hl.dot calls: q@k^T for attention scores, sim@v for local output
  - One hl.dot: q@h for state contribution
  - Causal mask via hl.arange comparison
  - hl.specialize(K), hl.specialize(V)
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1,  64,  2, 64,  64):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 128,  4, 64,  64):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (1, 256,  4, 64, 128):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    # Benchmark shapes
    (1,   64, 1, 64,  64):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2,  512, 3, 64,  64):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 1024, 3, 64,  64):  helion.Config(block_sizes=[], num_warps=8, num_stages=2),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[], num_warps=8, num_stages=2)

_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def chunk_fwd_o(
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]  (v_new — corrected values)
        h: torch.Tensor,     # [B, NT, H, K, V]  (per-chunk states)
        g: torch.Tensor,     # [B, T, H]  (cumulative gate)
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            # Load chunk data
            g_vals = g[b_idx, tile_t, h_idx]                  # [C]
            q_tile = q[b_idx, tile_t, h_idx, :]               # [C, K]
            k_tile = k[b_idx, tile_t, h_idx, :]               # [C, K]
            v_tile = v[b_idx, tile_t, h_idx, :]               # [C, V]

            # ── Intra-chunk: causal attention within the chunk ──────────
            # qk = q @ k^T : [C, K] @ [K, C] → [C, C]
            qk = hl.dot(q_tile, k_tile.T)

            # Causal mask + gated scores
            idx = hl.arange(tile_t.block_size)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)

            # local_out = sim @ v : [C, C] @ [C, V] → [C, V]
            local_out = hl.dot(sim.to(v.dtype), v_tile)

            # ── Inter-chunk: state contribution ──────────────────────────
            # q_scaled = q * exp(g) : [C, K]
            q_s = q_tile * torch.exp(g_vals)[:, None]

            # global_out = q_scaled @ h[b,c,h] : [C, K] @ [K, V] → [C, V]
            global_out = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])

            # ── Combine ──────────────────────────────────────────────────
            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return chunk_fwd_o

def _get_kernel(config: helion.Config):
    key = id(config)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]

def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    config = SHAPE_CONFIGS.get((B, T, H, K, V), FALLBACK_CONFIG)
    kernel = _get_kernel(config)
    return kernel(q, k, v_new, h, g, scale)
