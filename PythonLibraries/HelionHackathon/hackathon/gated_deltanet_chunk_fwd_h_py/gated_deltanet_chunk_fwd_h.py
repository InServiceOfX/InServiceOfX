#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius
"""
Gated DeltaNet chunk_fwd_h — Helion Submission
Team: luminous-kernels

Algorithm (inter-chunk state recurrence):
  For each (b, h) pair, h_state ∈ ℝ^{K×V} starts at 0. Sequentially over chunks c:
    1. Store:   h_out[b, c, h] = h_state
    2. Compute: v_new = u - w @ h_state           (project state → correction)
    3. Gate:    v_gated[t] = v_new[t] * exp(g[last] - g[t])
    4. Decay:   h_state *= exp(g[last])
    5. Update:  h_state += k^T @ v_gated           (rank-C outer product update)

Kernel design (follows Helion examples/gdn_fwd_h.py pattern):
  - hl.tile([B*H, V], block_size=[1, block_v]): parallel over (batch, head, V-slice)
  - Sequential hl.tile(T, block_size=C=64) over chunks (the recurrence is causal)
  - hl.specialize(K): bake key dimension into compiled kernel
  - hl.dot(w, h, ...) for w@h projection; hl.dot(k.T, v, acc=h) for fused rank update
  - state h lives in registers as [K, tile_v] — never spills to global memory between chunks
  - Zero inline_triton/asm (pure Helion DSL)

Fixes over baseline:
  - Baseline computes proj = (dot1 + dot2) * 0.5 and upd = (dot1 + dot2) * 0.5
    (redundant double-compute). We compute each dot once.
  - Baseline uses block_size=[1, 8]. We use hl.register_block_size(V) for wider V tiles.
  - Use hl.dot(..., acc=b_h) for fused accumulate (saves one add instruction).
"""

from task import input_t, output_t
import torch
import helion
import helion.language as hl

# ── Per-shape configs ──────────────────────────────────────────────────────────
# Keys: (B, T, H, K, V) — must cover all test + benchmark shapes from task.yml
# Configs are placeholder defaults; run autotune_chunk_fwd_h.py on Nebius to fill in.

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes — block_sizes=[V_tile] for the one variable hl.tile dim
    (1,  64,  2, 64,  64):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 128,  4, 64,  64):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (1, 256,  4, 64, 128):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    # Benchmark shapes
    (1,   64, 1, 64,  64):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2,  512, 3, 64,  64):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 1024, 3, 64,  64):  helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
}

FALLBACK_CONFIG = helion.Config(block_sizes=[64], num_warps=4, num_stages=1)


# ── Kernel factory ────────────────────────────────────────────────────────────
_kernel_cache: dict = {}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def chunk_fwd_h(
        k: torch.Tensor,   # [B, T, H, K] — keys
        w: torch.Tensor,   # [B, T, H, K] — WY-transformed keys
        u: torch.Tensor,   # [B, T, H, V] — WY-transformed values
        g: torch.Tensor,   # [B, T, H]    — cumulative gate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── Host code ──────────────────────────────────────────────────────
        B, T, H, K = k.shape
        V = u.shape[-1]
        K = hl.specialize(K)     # bake K as compile-time constant
        V = hl.specialize(V)     # bake V as compile-time constant
        C = 64                   # chunk size (fixed by problem spec)
        NT = (T + C - 1) // C   # number of chunks

        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)

        BH = B * H              # flatten batch × heads for parallelism

        # ── Device code ────────────────────────────────────────────────────
        # Tile over (batch*heads, V). Each thread block processes one (b,h) pair
        # and a slice of the V dimension. The chunk loop inside is sequential
        # (causal recurrence — chunks cannot be parallelized).
        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H   # batch index
            h_idx = flat.begin % H    # head index

            # Hidden state h ∈ ℝ^{K × tile_v}, lives in registers across chunks
            state = hl.zeros([K, tv], dtype=torch.float32)

            # Sequential recurrence over chunks c = 0, 1, ..., NT-1
            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1

                # 1. Store: h_out[b, c, h] = state
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                # 2. Compute: v_new = u - w @ state
                # w[b, tc, h, :] is [C, K]; state is [K, tile_v]
                # hl.dot computes [C, K] @ [K, tile_v] → [C, tile_v]
                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                # 3. Gate: v_gated[t] = v_new[t] * exp(g[last] - g[t])
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]

                # 4. Decay: state *= exp(g[last])
                state = state * torch.exp(g_end)

                # 5. Update: state += k^T @ v_gated  (fused accumulate)
                # k_adj.T is [K, C]; diff is [C, tile_v]
                # hl.dot with acc=state fuses the add: state = state + k^T @ v_gated
                state = hl.dot(k_adj.T, diff, acc=state)

        return h_out, v_out

    return chunk_fwd_h

def _get_kernel(config: helion.Config):
    key = id(config)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_kernel(config)
    return _kernel_cache[key]


# ── Entry point ───────────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    config = SHAPE_CONFIGS.get((B, T, H, K, V), FALLBACK_CONFIG)
    kernel = _get_kernel(config)
    return kernel(k, w, u, g)
