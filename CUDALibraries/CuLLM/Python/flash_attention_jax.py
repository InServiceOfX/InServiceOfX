"""
FlashAttention in JAX — manual tiled implementation + comparison with naive.

Demonstrates:
  1. Naive O(N²) attention (baseline, materializes full attention matrix)
  2. Manual FlashAttention in JAX using jax.lax.while_loop (tile-by-tile)
  3. JAX built-in dot_product_attention (for benchmarking)
  4. Timing comparison via jax.block_until_ready

Reference: Dao et al. (2022) FlashAttention, Algorithm 1.
See Documents/FlashAttention.md for the mathematical derivation.

Usage (inside the PropulsionWithCUDA Docker container):
    python3 Python/flash_attention_jax.py

Dependencies: jax[cuda13-local] (installed in the Docker image)
"""

import time
import jax
import jax.numpy as jnp
from functools import partial


# ── Naive attention ────────────────────────────────────────────────────────────

@jax.jit
def naive_attention(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Exact scaled dot-product attention. Materializes N×N attention matrix.

    Args:
        Q: [B, NH, N, d]
        K: [B, NH, N, d]
        V: [B, NH, N, d]

    Returns:
        O: [B, NH, N, d]
    """
    d = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(d).astype(Q.dtype)
    # S: [B, NH, N, N]
    S = jnp.einsum('bhid,bhjd->bhij', Q, K) * scale
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('bhij,bhjd->bhid', P, V)


# ── Manual FlashAttention in JAX ───────────────────────────────────────────────

def flash_attention_jax(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    block_size: int = 64,
) -> jnp.ndarray:
    """
    FlashAttention forward pass implemented in JAX using jax.lax.fori_loop.

    Avoids materializing the N×N attention matrix by iterating over K/V tiles.
    Memory usage: O(N * d) rather than O(N²).

    Each outer step i processes one Q-row tile (Q[i*Br:(i+1)*Br]).
    Each inner step j processes one K/V tile.

    Args:
        Q: [B, NH, N, d]
        K: [B, NH, N, d]
        V: [B, NH, N, d]
        block_size: tile size Br = Bc (same for simplicity)

    Returns:
        O: [B, NH, N, d]
    """
    B, NH, N, d = Q.shape
    Br = block_size
    Bc = block_size
    scale = (1.0 / jnp.sqrt(d)).astype(Q.dtype)

    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    # Pad sequence dimension to multiple of block_size for clean slicing.
    N_pad = Tr * Br
    pad_len = N_pad - N
    if pad_len > 0:
        Q = jnp.pad(Q, ((0,0),(0,0),(0,pad_len),(0,0)))
        K = jnp.pad(K, ((0,0),(0,0),(0,pad_len),(0,0)))
        V = jnp.pad(V, ((0,0),(0,0),(0,pad_len),(0,0)))

    # Reshape into tiles: [B, NH, Tr, Br, d] and [B, NH, Tc, Bc, d]
    Q_tiles = Q.reshape(B, NH, Tr, Br, d)
    K_tiles = K.reshape(B, NH, Tc, Bc, d)
    V_tiles = V.reshape(B, NH, Tc, Bc, d)

    def process_row_tile(i, O_acc):
        """Process Q tile i, accumulating into O_acc [B, NH, Tr, Br, d]."""
        Q_i = Q_tiles[:, :, i, :, :]  # [B, NH, Br, d]

        # Per-tile running accumulators.
        m_i = jnp.full((B, NH, Br), -jnp.inf, dtype=Q.dtype)
        l_i = jnp.zeros((B, NH, Br), dtype=Q.dtype)
        O_i = jnp.zeros((B, NH, Br, d), dtype=Q.dtype)

        def process_col_tile(j, carry):
            m_i, l_i, O_i = carry
            K_j = K_tiles[:, :, j, :, :]  # [B, NH, Bc, d]
            V_j = V_tiles[:, :, j, :, :]  # [B, NH, Bc, d]

            # S_ij: [B, NH, Br, Bc]
            S_ij = jnp.einsum('bhid,bhjd->bhij', Q_i, K_j) * scale

            # Tile-local max and softmax numerators.
            m_ij = S_ij.max(axis=-1)                        # [B, NH, Br]
            P_ij = jnp.exp(S_ij - m_ij[..., None])         # [B, NH, Br, Bc]
            l_ij = P_ij.sum(axis=-1)                        # [B, NH, Br]

            # Online softmax: merge tile stats with running stats.
            m_new = jnp.maximum(m_i, m_ij)
            alpha = jnp.exp(m_i - m_new)                    # [B, NH, Br]
            beta  = jnp.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            # Update output: rescale old O_i, add new P_ij @ V_j.
            # O_i maintained as unnorm-rescaled accumulator, divided at end.
            pv = jnp.einsum('bhij,bhjd->bhid', P_ij, V_j)  # [B, NH, Br, d]
            O_new = (
                alpha[..., None] * l_i[..., None] * O_i
                + beta[..., None] * pv
            ) / l_new[..., None]

            return m_new, l_new, O_new

        m_i, l_i, O_i = jax.lax.fori_loop(0, Tc, process_col_tile, (m_i, l_i, O_i))

        # Write O_i into the output accumulator at row block i.
        return O_acc.at[:, :, i, :, :].set(O_i)

    O_tiles = jnp.zeros((B, NH, Tr, Br, d), dtype=Q.dtype)
    O_tiles = jax.lax.fori_loop(0, Tr, process_row_tile, O_tiles)

    # Reshape back and strip padding.
    O = O_tiles.reshape(B, NH, N_pad, d)
    return O[:, :, :N, :]


flash_attention_jax_jit = jax.jit(flash_attention_jax, static_argnames=['block_size'])


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(fn, *args, warmup: int = 3, repeats: int = 10, label: str = '') -> float:
    """Run fn(*args) warmup times, then time repeats iterations."""
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    avg_ms = 1e3 * sum(times) / len(times)
    print(f"  {label:<40s}  avg {avg_ms:.2f} ms")
    return avg_ms


def verify_outputs_match(
    O_ref: jnp.ndarray,
    O_flash: jnp.ndarray,
    atol: float = 1e-3,
    label: str = '',
) -> None:
    max_err = float(jnp.max(jnp.abs(O_ref - O_flash)))
    match = max_err < atol
    status = 'PASS' if match else 'FAIL'
    print(f"  [{status}] {label} — max |err| = {max_err:.2e} (atol={atol})")


def main() -> None:
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}\n")

    key = jax.random.PRNGKey(0)

    configs = [
        (1, 1,  128, 64,  "B=1 NH=1  N=128  d=64"),
        (1, 4,  512, 64,  "B=1 NH=4  N=512  d=64"),
        (2, 8, 1024, 64,  "B=2 NH=8  N=1024 d=64"),
    ]

    for B, NH, N, d, label in configs:
        print(f"Config: {label}")
        shape = (B, NH, N, d)
        Q = jax.random.normal(key, shape)
        K = jax.random.normal(key, shape)
        V = jax.random.normal(key, shape)

        O_naive = naive_attention(Q, K, V)
        jax.block_until_ready(O_naive)

        O_flash = flash_attention_jax_jit(Q, K, V, block_size=64)
        jax.block_until_ready(O_flash)

        verify_outputs_match(O_naive, O_flash, atol=1e-3, label="FlashAttn vs naive")

        benchmark(naive_attention, Q, K, V, label="naive_attention")
        benchmark(flash_attention_jax_jit, Q, K, V, label="flash_attention_jax")

        try:
            builtin_fn = jax.jit(
                lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=False)
            )
            O_builtin = builtin_fn(Q, K, V)
            jax.block_until_ready(O_builtin)
            verify_outputs_match(O_naive, O_builtin, atol=1e-3, label="jax builtin vs naive")
            benchmark(builtin_fn, Q, K, V, label="jax.nn.dot_product_attention")
        except AttributeError:
            print("  jax.nn.dot_product_attention not available in this JAX version")

        print()


if __name__ == '__main__':
    main()
