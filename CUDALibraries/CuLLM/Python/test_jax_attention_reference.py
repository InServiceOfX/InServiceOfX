"""Tests for the JAX attention references.

Run inside the PropulsionWithCUDA container:

    cd /InServiceOfX
    python3 -m pytest CUDALibraries/CuLLM/Python/test_jax_attention_reference.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jax_attention_reference import (
    flash_attention,
    flash_attention2,
    jax_builtin_attention,
    make_cullm_inputs,
    max_abs_error,
    online_normalizer,
    online_softmax,
    standard_attention,
)


def test_online_softmax_matches_jax_softmax() -> None:
    row = jnp.asarray([3.0, -7.0, 0.5, 11.0, -2.0, 9.0], dtype=jnp.float32)
    max_value, denominator, logsumexp = online_normalizer(row)
    expected_logsumexp = jax.nn.logsumexp(row)

    assert float(jnp.abs(logsumexp - expected_logsumexp)) < 1e-6
    assert float(jnp.abs(max_value - jnp.max(row))) < 1e-6
    assert float(jnp.abs(denominator - jnp.sum(jnp.exp(row - max_value)))) < 1e-6
    assert max_abs_error(online_softmax(row), jax.nn.softmax(row)) < 1e-6


def _check_attention_case(
    sequence_length: int,
    head_dim: int,
    causal: bool,
    block_size: int,
    heads: int = 1,
) -> None:
    q, k, v = make_cullm_inputs(1, heads, sequence_length, head_dim)
    reference = standard_attention(q, k, v, causal=causal)
    eager = flash_attention(q, k, v, block_size=block_size, causal=causal)
    delayed, logsumexp = flash_attention2(
        q,
        k,
        v,
        block_size=block_size,
        causal=causal,
        return_logsumexp=True,
    )

    # These are different float32 realizations of the same map: standard
    # attention performs one full reduction, while tiled attention merges
    # partial reductions.  The association order changes, so use a realistic
    # float32 cross-realization tolerance rather than a bitwise-style one.
    assert max_abs_error(reference, eager) < 3e-4
    assert max_abs_error(reference, delayed) < 3e-4

    scores = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(float(head_dim))
    if causal:
        rows = jnp.arange(sequence_length)[:, None]
        cols = jnp.arange(sequence_length)[None, :]
        scores = jnp.where(cols <= rows, scores, -jnp.inf)
    expected_lse = jax.nn.logsumexp(scores, axis=-1)
    assert max_abs_error(logsumexp, expected_lse) < 3e-4


def test_flash_attention_matches_standard_non_multiple_tiles() -> None:
    _check_attention_case(
        sequence_length=100,
        head_dim=32,
        causal=False,
        block_size=32,
    )


def test_flash_attention2_matches_standard_causal() -> None:
    _check_attention_case(
        sequence_length=150,
        head_dim=64,
        causal=True,
        block_size=64,
    )


def test_multihead_matches_standard() -> None:
    _check_attention_case(
        sequence_length=96,
        head_dim=32,
        causal=False,
        block_size=32,
        heads=4,
    )


def test_builtin_xla_wrapper_matches_standard() -> None:
    q, k, v = make_cullm_inputs(1, 2, 64, 32)
    reference = standard_attention(q, k, v, causal=True)
    builtin = jax.jit(
        functools.partial(
            jax_builtin_attention,
            causal=True,
            implementation="xla",
        )
    )(q, k, v)
    assert max_abs_error(reference, builtin) < 2e-5
