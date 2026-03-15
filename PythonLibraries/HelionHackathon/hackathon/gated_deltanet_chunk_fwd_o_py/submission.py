from task import input_t, output_t

import torch
import helion
import helion.language as hl

CHUNK_SIZE = 64

# Start conservative. We'll replace benchmark entries with autotuned configs from Nebius.
SHAPE_CONFIGS: dict[tuple[int, int, int, int, int], helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[], num_warps=8, num_stages=2)

_kernel_cache: dict[tuple, object] = {}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = CHUNK_SIZE
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_tile = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_tile = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_tile = v[b_idx, tile_t, h_idx, :].to(torch.float32)

            qk = hl.dot(q_tile, k_tile.T, out_dtype=torch.float32)
            idx = hl.arange(tile_t.block_size)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)
            local_out = hl.dot(sim, v_tile, out_dtype=torch.float32)

            q_gated = q_tile * torch.exp(g_vals)[:, None]
            global_out = hl.dot(q_gated, h[b_idx, c_idx, h_idx, :, :].to(torch.float32), out_dtype=torch.float32)

            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


def _get_kernel(config: helion.Config):
    key = (tuple(config.block_sizes), config.num_warps, config.num_stages)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    config = SHAPE_CONFIGS.get((B, T, H, K, V), FALLBACK_CONFIG)
    kernel = _get_kernel(config)
    return kernel(q, k, v_new, h, g, K ** -0.5)
