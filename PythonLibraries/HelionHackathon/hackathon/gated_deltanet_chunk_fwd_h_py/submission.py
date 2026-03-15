from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # ---- Autotuned configs from B200 (best min latency) ----

    # (1,64,2,64,64) — 0.0061 ms
    (1, 64, 2, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['', 'first', 'first', '', ''],
        loop_orders=[[0, 1]],
        num_stages=5,
        num_warps=1,
        pid_type='flat',
    ),

    # (2,128,4,64,64) — 0.0144 ms
    (2, 128, 4, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'],
        l2_groupings=[8],
        load_eviction_policies=['last', '', 'first', 'first', 'last'],
        loop_orders=[[0, 1]],
        num_stages=1,
        num_warps=16,
        pid_type='flat',
    ),

    # (1,256,4,64,128) — 0.0327 ms
    (1, 256, 4, 64, 128): helion.Config(
        block_sizes=[],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'],
        l2_groupings=[1],
        load_eviction_policies=['first', 'last', '', 'first', 'last'],
        loop_orders=[[0, 1]],
        num_stages=1,
        num_warps=2,
        pid_type='flat',
    ),

    # (1,64,1,64,64) — 0.0062 ms
    (1, 64, 1, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        l2_groupings=[4],
        load_eviction_policies=['first', '', 'last', 'last', 'last'],
        loop_orders=[[0, 1]],
        num_stages=3,
        num_warps=1,
        pid_type='flat',
    ),

    # (2,512,3,64,64) — not tuned, using conservative config
    (2, 512, 3, 64, 64): helion.Config(
        block_sizes=[],
        num_stages=1,
        num_warps=8,
        pid_type='flat',
    ),

    # (2,1024,3,64,64) — not tuned, using conservative config
    (2, 1024, 3, 64, 64): helion.Config(
        block_sizes=[],
        num_stages=1,
        num_warps=8,
        pid_type='flat',
    ),

    # Ranked shapes — conservative fallbacks
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (4, 2048, 8, 64, 64):   helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat')


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        k: torch.Tensor,   # [B, T, H, K]
        w: torch.Tensor,   # [B, T, H, K]
        u: torch.Tensor,   # [B, T, H, V]
        g: torch.Tensor,   # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        NT = T // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)

        BH = B * H

        for flat_bh, tile_v in hl.tile([BH, V]):
            b = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            state = hl.zeros([K, tile_v], dtype=torch.float32)

            for tile_t in hl.tile(T, block_size=C):
                c_idx = tile_t.begin // C

                # 1. Store current state before update
                h_out[b, c_idx, h_idx, :, tile_v] = state.to(k.dtype)

                # 2. v_new = u - w @ state
                w_chunk = w[b, tile_t, h_idx, :].to(torch.float32)       # [C, K]
                u_chunk = u[b, tile_t, h_idx, tile_v].to(torch.float32)   # [C, BV]
                proj = hl.dot(w_chunk, state, out_dtype=torch.float32)    # [C, BV]
                v_new = u_chunk - proj
                v_out[b, tile_t, h_idx, tile_v] = v_new.to(u.dtype)

                # 3. Extract g_last = g[last timestep in chunk]
                #    Use masked sum to avoid symbolic indexing issues
                g_chunk = g[b, tile_t, h_idx].to(torch.float32)          # [C]
                t_idx = hl.arange(tile_t.block_size)
                last_mask = (t_idx == (C - 1)).to(torch.float32)
                g_last = (g_chunk * last_mask).sum()                      # scalar

                # 4. gate[t] = exp(g_last - g[t])
                gate = torch.exp(g_last - g_chunk)                        # [C]
                v_gated = v_new * gate[:, None]                           # [C, BV]

                # 5. State update: h = h * exp(g_last) + k^T @ v_gated
                k_chunk = k[b, tile_t, h_idx, :].to(torch.float32)       # [C, K]
                decay = torch.exp(g_last)
                state = state * decay + hl.dot(
                    k_chunk.T, v_gated, out_dtype=torch.float32
                )

        return h_out, v_out

    return kernel


_kernel_cache: dict[tuple, object] = {}


def _get_kernel(shape_key):
    if shape_key not in _kernel_cache:
        config = SHAPE_CONFIGS.get(shape_key, FALLBACK_CONFIG)
        _kernel_cache[shape_key] = _make_kernel(config)
    return _kernel_cache[shape_key]


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _get_kernel((B, T, H, K, V))
    return kernel(k, w, u, g)