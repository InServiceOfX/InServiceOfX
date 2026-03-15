#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius
"""
Gated DeltaNet recompute_w_u — Helion Submission
Team: luminous-kernels

Algorithm (WY-transform forward):
  Sequence divided into chunks of C=64 timesteps. For each chunk independently:
    u = A @ diag(beta) @ v   = A @ (v * beta[:, None])
    w = A @ diag(beta*exp(g)) @ k = A @ (k * (beta * exp(g))[:, None])

  Where A is a [C, C] lower-triangular WY matrix per (b, chunk, h).

Kernel design:
  - hl.tile([B*H, T], block_size=[1, C]): each tile = one chunk of one (b,h)
  - The inner computation is two matmuls: A @ scaled_v and A @ scaled_k
  - hl.dot for both matmuls (baseline loops element-by-element, TWICE)
  - hl.specialize(K), hl.specialize(V), hl.specialize(C)

Fixes over baseline:
  - Baseline does element-by-element loop for ci in range(C) TWICE (forward + backward)
    and averages them. We use two hl.dot matmuls — O(1) calls vs O(C).
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1,  64,  2, 64,  64):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
    (2, 128,  4, 64,  64):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
    (1, 256,  4, 64, 128):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
    # Benchmark shapes
    (1,   64, 1, 64,  64):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
    (2,  512, 3, 64,  64):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
    (2, 1024, 3, 64,  64):  helion.Config(block_sizes=[], num_warps=4, num_stages=1),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[], num_warps=4, num_stages=1)

_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def recompute_w_u(
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        beta: torch.Tensor,  # [B, T, H]
        A: torch.Tensor,     # [B, T, H, BT]  BT=C=64
        g: torch.Tensor,     # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])   # chunk size = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)

        BH = B * H

        # Tile over (batch*heads, T). Each tile = one chunk of C timesteps.
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            # Load per-chunk data
            # A_chunk: [C, C] — the WY matrix for this chunk
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)  # [C, C]

            # beta and g for this chunk
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)  # [C]
            g_chunk = g[b_idx, rt, h_idx].to(torch.float32)       # [C]

            # k and v for this chunk
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)    # [C, K]
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)    # [C, V]

            # Scale: v_scaled = v * beta[:, None]
            v_scaled = v_chunk * beta_chunk[:, None]               # [C, V]

            # Scale: k_scaled = k * (beta * exp(g))[:, None]
            k_scaled = k_chunk * (beta_chunk * torch.exp(g_chunk))[:, None]  # [C, K]

            # u = A @ v_scaled   → [C, C] @ [C, V] = [C, V]
            u_chunk = hl.dot(A_chunk, v_scaled, out_dtype=torch.float32)

            # w = A @ k_scaled   → [C, C] @ [C, K] = [C, K]
            w_chunk = hl.dot(A_chunk, k_scaled, out_dtype=torch.float32)

            w_out[b_idx, rt, h_idx, :] = w_chunk.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_chunk.to(v.dtype)

        return w_out, u_out

    return recompute_w_u

def _get_kernel(config: helion.Config):
    key = id(config)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]

def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    config = SHAPE_CONFIGS.get((B, T, H, K, V), FALLBACK_CONFIG)
    kernel = _get_kernel(config)
    return kernel(k, v, beta, A, g)
