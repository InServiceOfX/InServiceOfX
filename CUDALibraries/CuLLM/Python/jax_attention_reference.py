"""JAX reference implementations for CuLLM attention kernels.

The functions here are correctness and benchmarking references, not an attempt
to beat XLA/cuDNN.  Tensor layout follows CuLLM's logical multi-head layout:

    (batch, heads, sequence, head_dim)

Implemented variants:
  * standard_attention: materializes the full N x N attention matrix.
  * online_softmax: one-row online normalizer reference.
  * flash_attention: tiled exact attention with eager normalized output.
  * flash_attention2: tiled exact attention with delayed normalization and
    optional logsumexp output, matching the FlashAttention-2 forward math.
  * jax_builtin_attention: wrapper around jax.nn.dot_product_attention.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


Array = jax.Array


def cullm_deterministic_values(count: int, seed: int) -> Array:
    """Match the deterministic input generator used by CuLLM C++ benchmarks."""
    i = jnp.arange(count, dtype=jnp.int32)
    values = ((i * seed + 7 * (i % 13)) % 211 - 105).astype(jnp.float32)
    return values / jnp.float32(105.0)


def make_cullm_inputs(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
) -> tuple[Array, Array, Array]:
    """Return Q, K, V with the same flattened values as the C++ tests."""
    shape = (batch_size, num_heads, sequence_length, head_dim)
    count = batch_size * num_heads * sequence_length * head_dim
    q = cullm_deterministic_values(count, 3).reshape(shape)
    k = cullm_deterministic_values(count, 5).reshape(shape)
    v = cullm_deterministic_values(count, 11).reshape(shape)
    return q, k, v


def _causal_mask(sequence_length: int) -> Array:
    rows = jnp.arange(sequence_length)[:, None]
    cols = jnp.arange(sequence_length)[None, :]
    return cols <= rows


@partial(jax.jit, static_argnames=("causal",))
def standard_attention(q: Array, k: Array, v: Array, causal: bool = False) -> Array:
    """Exact scaled dot-product attention, materializing scores and weights."""
    head_dim = q.shape[-1]
    scale = jnp.asarray(1.0 / (head_dim ** 0.5), dtype=q.dtype)
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    if causal:
        mask = _causal_mask(q.shape[-2])
        scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhij,bhjd->bhid", weights, v)


@jax.jit
def online_normalizer(row: Array) -> tuple[Array, Array, Array]:
    """Return (m, l, logsumexp) using the online softmax recurrence."""

    def step(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        max_value, denominator = carry
        x = row[index]
        new_max = jnp.maximum(max_value, x)
        new_denominator = (
            jnp.exp(max_value - new_max) * denominator + jnp.exp(x - new_max)
        )
        return new_max, new_denominator

    init = (
        jnp.asarray(-jnp.inf, dtype=row.dtype),
        jnp.asarray(0.0, dtype=row.dtype),
    )
    max_value, denominator = jax.lax.fori_loop(0, row.shape[0], step, init)
    return max_value, denominator, max_value + jnp.log(denominator)


@jax.jit
def online_softmax(row: Array) -> Array:
    """Softmax computed from the online logsumexp statistic."""
    _, _, logsumexp = online_normalizer(row)
    return jnp.exp(row - logsumexp)


def _pad_sequence(x: Array, block_size: int) -> tuple[Array, int]:
    sequence_length = x.shape[-2]
    blocks = (sequence_length + block_size - 1) // block_size
    padded_length = blocks * block_size
    pad = padded_length - sequence_length
    if pad == 0:
        return x, padded_length
    return jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))), padded_length


def _tile_mask(
    row_start: int,
    col_start: int,
    row_block: int,
    col_block: int,
    sequence_length: int,
    causal: bool,
) -> Array:
    query_indices = row_start + jnp.arange(row_block)
    key_indices = col_start + jnp.arange(col_block)
    valid = (query_indices[:, None] < sequence_length) & (
        key_indices[None, :] < sequence_length
    )
    if causal:
        valid = valid & (key_indices[None, :] <= query_indices[:, None])
    return valid


@partial(jax.jit, static_argnames=("block_size", "causal"))
def flash_attention(
    q: Array,
    k: Array,
    v: Array,
    block_size: int = 64,
    causal: bool = False,
) -> Array:
    """Tiled exact attention with eager normalized output after every tile."""
    batch_size, num_heads, sequence_length, head_dim = q.shape
    q, padded_length = _pad_sequence(q, block_size)
    k, _ = _pad_sequence(k, block_size)
    v, _ = _pad_sequence(v, block_size)

    row_blocks = padded_length // block_size
    col_blocks = row_blocks
    scale = jnp.asarray(1.0 / (head_dim ** 0.5), dtype=q.dtype)

    q_tiles = q.reshape(batch_size, num_heads, row_blocks, block_size, head_dim)
    k_tiles = k.reshape(batch_size, num_heads, col_blocks, block_size, head_dim)
    v_tiles = v.reshape(batch_size, num_heads, col_blocks, block_size, head_dim)

    def process_row_tile(row_tile: int, output_tiles: Array) -> Array:
        q_i = q_tiles[:, :, row_tile, :, :]
        row_start = row_tile * block_size
        max_i = jnp.full((batch_size, num_heads, block_size), -jnp.inf, q.dtype)
        denom_i = jnp.zeros((batch_size, num_heads, block_size), q.dtype)
        output_i = jnp.zeros((batch_size, num_heads, block_size, head_dim), q.dtype)

        def process_col_tile(
            col_tile: int, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            max_i, denom_i, output_i = carry
            k_j = k_tiles[:, :, col_tile, :, :]
            v_j = v_tiles[:, :, col_tile, :, :]
            col_start = col_tile * block_size
            mask = _tile_mask(
                row_start,
                col_start,
                block_size,
                block_size,
                sequence_length,
                causal,
            )

            scores = jnp.einsum("bhid,bhjd->bhij", q_i, k_j) * scale
            masked_scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
            tile_max = jnp.max(masked_scores, axis=-1)
            new_max = jnp.maximum(max_i, tile_max)
            safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, 0)
            old_scale = jnp.where(
                denom_i > 0, jnp.exp(max_i - safe_new_max), 0
            )
            weights = jnp.where(
                mask[None, None, :, :],
                jnp.exp(scores - safe_new_max[..., None]),
                0,
            )
            tile_denom = jnp.sum(weights, axis=-1)
            new_denom = old_scale * denom_i + tile_denom
            tile_output = jnp.einsum("bhij,bhjd->bhid", weights, v_j)
            numerator = old_scale[..., None] * denom_i[..., None] * output_i
            numerator = numerator + tile_output
            new_output = jnp.where(
                new_denom[..., None] > 0, numerator / new_denom[..., None], 0
            )
            return new_max, new_denom, new_output

        _, _, output_i = jax.lax.fori_loop(
            0, col_blocks, process_col_tile, (max_i, denom_i, output_i)
        )
        return output_tiles.at[:, :, row_tile, :, :].set(output_i)

    output_tiles = jnp.zeros(
        (batch_size, num_heads, row_blocks, block_size, head_dim), dtype=q.dtype
    )
    output_tiles = jax.lax.fori_loop(0, row_blocks, process_row_tile, output_tiles)
    output = output_tiles.reshape(batch_size, num_heads, padded_length, head_dim)
    return output[:, :, :sequence_length, :]


@partial(jax.jit, static_argnames=("block_size", "causal", "return_logsumexp"))
def flash_attention2(
    q: Array,
    k: Array,
    v: Array,
    block_size: int = 64,
    causal: bool = False,
    return_logsumexp: bool = False,
) -> Array | tuple[Array, Array]:
    """Tiled exact attention with FA-2 delayed normalization.

    The running output is the unnormalized accumulator.  Division by the online
    denominator happens once in the epilogue.  If requested, the function also
    returns L = m + log(l), the statistic needed by the backward pass.
    """
    batch_size, num_heads, sequence_length, head_dim = q.shape
    q, padded_length = _pad_sequence(q, block_size)
    k, _ = _pad_sequence(k, block_size)
    v, _ = _pad_sequence(v, block_size)

    row_blocks = padded_length // block_size
    col_blocks = row_blocks
    scale = jnp.asarray(1.0 / (head_dim ** 0.5), dtype=q.dtype)

    q_tiles = q.reshape(batch_size, num_heads, row_blocks, block_size, head_dim)
    k_tiles = k.reshape(batch_size, num_heads, col_blocks, block_size, head_dim)
    v_tiles = v.reshape(batch_size, num_heads, col_blocks, block_size, head_dim)

    def process_row_tile(
        row_tile: int, carry: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        output_tiles, logsumexp_tiles = carry
        q_i = q_tiles[:, :, row_tile, :, :]
        row_start = row_tile * block_size
        max_i = jnp.full((batch_size, num_heads, block_size), -jnp.inf, q.dtype)
        denom_i = jnp.zeros((batch_size, num_heads, block_size), q.dtype)
        output_i = jnp.zeros((batch_size, num_heads, block_size, head_dim), q.dtype)

        def process_col_tile(
            col_tile: int, inner: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            max_i, denom_i, output_i = inner
            k_j = k_tiles[:, :, col_tile, :, :]
            v_j = v_tiles[:, :, col_tile, :, :]
            col_start = col_tile * block_size
            mask = _tile_mask(
                row_start,
                col_start,
                block_size,
                block_size,
                sequence_length,
                causal,
            )

            scores = jnp.einsum("bhid,bhjd->bhij", q_i, k_j) * scale
            masked_scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
            tile_max = jnp.max(masked_scores, axis=-1)
            new_max = jnp.maximum(max_i, tile_max)
            safe_new_max = jnp.where(jnp.isfinite(new_max), new_max, 0)
            old_scale = jnp.where(
                denom_i > 0, jnp.exp(max_i - safe_new_max), 0
            )
            weights = jnp.where(
                mask[None, None, :, :],
                jnp.exp(scores - safe_new_max[..., None]),
                0,
            )
            tile_denom = jnp.sum(weights, axis=-1)
            new_denom = old_scale * denom_i + tile_denom
            tile_output = jnp.einsum("bhij,bhjd->bhid", weights, v_j)
            new_output = old_scale[..., None] * output_i + tile_output
            return new_max, new_denom, new_output

        max_i, denom_i, output_i = jax.lax.fori_loop(
            0, col_blocks, process_col_tile, (max_i, denom_i, output_i)
        )
        normalized = jnp.where(denom_i[..., None] > 0, output_i / denom_i[..., None], 0)
        logsumexp_i = jnp.where(denom_i > 0, max_i + jnp.log(denom_i), -jnp.inf)
        output_tiles = output_tiles.at[:, :, row_tile, :, :].set(normalized)
        logsumexp_tiles = logsumexp_tiles.at[:, :, row_tile, :].set(logsumexp_i)
        return output_tiles, logsumexp_tiles

    output_tiles = jnp.zeros(
        (batch_size, num_heads, row_blocks, block_size, head_dim), dtype=q.dtype
    )
    logsumexp_tiles = jnp.full(
        (batch_size, num_heads, row_blocks, block_size), -jnp.inf, dtype=q.dtype
    )
    output_tiles, logsumexp_tiles = jax.lax.fori_loop(
        0, row_blocks, process_row_tile, (output_tiles, logsumexp_tiles)
    )
    output = output_tiles.reshape(batch_size, num_heads, padded_length, head_dim)
    output = output[:, :, :sequence_length, :]
    logsumexp = logsumexp_tiles.reshape(batch_size, num_heads, padded_length)
    logsumexp = logsumexp[:, :, :sequence_length]
    if return_logsumexp:
        return output, logsumexp
    return output


def jax_builtin_attention(
    q: Array,
    k: Array,
    v: Array,
    causal: bool = False,
    implementation: str | None = None,
) -> Array:
    """Call jax.nn.dot_product_attention and convert layout back to CuLLM."""
    q_btnh = jnp.transpose(q, (0, 2, 1, 3))
    k_bsnh = jnp.transpose(k, (0, 2, 1, 3))
    v_bsnh = jnp.transpose(v, (0, 2, 1, 3))
    output = jax.nn.dot_product_attention(
        q_btnh,
        k_bsnh,
        v_bsnh,
        is_causal=causal,
        implementation=implementation,
    )
    return jnp.transpose(output, (0, 2, 1, 3))


def benchmark(
    fn: Callable[[], Array],
    warmups: int = 3,
    repeats: int = 20,
) -> float:
    """Return mean milliseconds for a zero-argument JAX callable."""
    for _ in range(warmups):
        jax.block_until_ready(fn())
    start = jax.default_backend()  # force backend initialization before timing
    del start

    import time

    elapsed = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        elapsed += time.perf_counter() - t0
    return 1000.0 * elapsed / repeats


def max_abs_error(a: Array, b: Array) -> float:
    return float(jnp.max(jnp.abs(a - b)))


def main() -> None:
    print(f"JAX {jax.__version__}")
    print(f"devices: {jax.devices()}")
    print()
    print(
        f"{'B':>2} {'H':>2} {'N':>5} {'d':>3} {'causal':>6} "
        f"{'standard':>10} {'FA':>10} {'FA-2':>10} {'builtin-xla':>12} "
        f"{'err FA-2':>10}"
    )
    for batch_size, heads, sequence_length, head_dim, causal in (
        (1, 1, 128, 32, False),
        (1, 1, 192, 64, False),
        (1, 1, 192, 64, True),
        (1, 4, 512, 64, False),
    ):
        q, k, v = make_cullm_inputs(batch_size, heads, sequence_length, head_dim)
        reference = standard_attention(q, k, v, causal=causal)
        fa = partial(flash_attention, q, k, v, block_size=64, causal=causal)
        fa2 = partial(flash_attention2, q, k, v, block_size=64, causal=causal)
        builtin = partial(jax_builtin_attention, q, k, v, causal=causal, implementation="xla")
        fa2_output = fa2()
        print(
            f"{batch_size:2d} {heads:2d} {sequence_length:5d} {head_dim:3d} "
            f"{str(causal):>6} "
            f"{benchmark(partial(standard_attention, q, k, v, causal=causal)):10.3f} "
            f"{benchmark(fa):10.3f} "
            f"{benchmark(fa2):10.3f} "
            f"{benchmark(builtin):12.3f} "
            f"{max_abs_error(reference, fa2_output):10.2e}"
        )


if __name__ == "__main__":
    main()
