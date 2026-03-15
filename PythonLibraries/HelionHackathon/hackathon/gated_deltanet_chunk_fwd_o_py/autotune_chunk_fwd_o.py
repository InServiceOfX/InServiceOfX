import os
os.environ.setdefault("HELION_PRINT_REPRO", "1")

import torch
import helion
import helion.language as hl

CHUNK_SIZE = 64
BENCHMARK_SHAPES = [
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
]


def make_inputs(B, T, H, K, V, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / K**0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    h = torch.randn(B, T // CHUNK_SIZE, H, K, V, dtype=torch.float32, device=device)
    g_inc = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = g_inc.cumsum(dim=1)
    return q, k, v, h, g, K ** -0.5


@helion.kernel(static_shapes=True, autotune_effort="quick", print_repro=True, dot_precision="ieee")
def chunk_fwd_o_autotune(
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


for shape in BENCHMARK_SHAPES:
    print(f"\n=== AUTOTUNING {shape} ===")
    args = make_inputs(*shape, seed=123)
    out = chunk_fwd_o_autotune(*args)
    torch.cuda.synchronize()
    print(f"Finished {shape}: {tuple(out.shape)}")
