"""
test_fp8_group_quant.py — pytest suite for the FP8 per-group quantization kernel.

Run on Nebius B200:
    source ~/helion_env/bin/activate
    cd ~/reference-kernels/problems/helion/fp8_quant_py
    pytest test_fp8_group_quant.py -v

Tests are organised in three layers matching Ernest's style:
  1. Reference correctness  — does our output match the pure-PyTorch reference?
  2. Mathematical invariants — do the outputs satisfy the algebraic properties?
  3. Hackathon shapes        — all 5 test + 3 benchmark shapes from task.yml.
"""

import math
import pytest
import torch

# ── Guard: skip entire module if no CUDA ────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required",
)

# ── Bring in the kernel and reference under test ─────────────────────────────
# When run from the fp8_quant_py/ directory (as shown in the header),
# task.py, reference.py, and fp8_group_quant.py are all siblings.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import from whichever filename is present (submission.py on Nebius eval,
# fp8_group_quant.py in the local repo).
try:
    from fp8_group_quant import custom_kernel
except ImportError:
    from submission import custom_kernel  # type: ignore

from reference import ref_kernel, generate_input, FP8_MAX, FP8_MIN, FP8_EPS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_input(num_tokens: int, hidden_dim: int, group_size: int, seed: int = 42):
    """Generate the (x, x_q, x_s) input tuple on CUDA."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x   = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device="cuda",
                      generator=gen).contiguous()
    x_q = torch.empty(num_tokens, hidden_dim,            dtype=torch.float32, device="cuda")
    x_s = torch.empty(num_tokens, hidden_dim // group_size, dtype=torch.float32, device="cuda")
    return x, x_q, x_s


def reference_outputs(num_tokens, hidden_dim, group_size, seed=42):
    """Run the pure-PyTorch reference and return (q, s)."""
    data = make_input(num_tokens, hidden_dim, group_size, seed)
    return ref_kernel(data)


def kernel_outputs(num_tokens, hidden_dim, group_size, seed=42):
    """Run our Helion custom_kernel and return (q, s)."""
    data = make_input(num_tokens, hidden_dim, group_size, seed)
    return custom_kernel(data)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Correctness: hackathon test shapes (task.yml §tests)
# ─────────────────────────────────────────────────────────────────────────────

HACKATHON_TEST_SHAPES = [
    # (num_tokens, hidden_dim, group_size, seed)  — from task.yml
    (1,   256,   64,  4242),
    (4,   512,  128,  5236),
    (16, 1024,   64,  1001),
    (1,  4096,  128,  5531),
    (8,  4096,  128,  9173),
]

@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES,
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES])
def test_correctness_against_reference(T, H, G, seed):
    """Custom kernel matches reference within rtol=1e-3, atol=1e-3."""
    ref_q, ref_s = reference_outputs(T, H, G, seed)
    got_q, got_s = kernel_outputs(T, H, G, seed)

    assert torch.allclose(got_q, ref_q, rtol=1e-3, atol=1e-3), (
        f"Quantized values mismatch for ({T},{H},{G}): "
        f"max_diff={(got_q - ref_q).abs().max().item():.4e}"
    )
    assert torch.allclose(got_s, ref_s, rtol=1e-3, atol=1e-3), (
        f"Scale values mismatch for ({T},{H},{G}): "
        f"max_diff={(got_s - ref_s).abs().max().item():.4e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correctness: hackathon benchmark shapes (task.yml §benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

HACKATHON_BENCHMARK_SHAPES = [
    (256,  4096, 128, 2146),
    (256,  8192, 128, 3129),
    (4096, 7168, 128, 54352),
]

@pytest.mark.parametrize("T,H,G,seed", HACKATHON_BENCHMARK_SHAPES,
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_BENCHMARK_SHAPES])
def test_correctness_benchmark_shapes(T, H, G, seed):
    """Kernel is correct on all benchmark shapes too."""
    ref_q, ref_s = reference_outputs(T, H, G, seed)
    got_q, got_s = kernel_outputs(T, H, G, seed)

    assert torch.allclose(got_q, ref_q, rtol=1e-3, atol=1e-3), (
        f"Benchmark shape ({T},{H},{G}): "
        f"max_diff={(got_q - ref_q).abs().max().item():.4e}"
    )
    assert torch.allclose(got_s, ref_s, rtol=1e-3, atol=1e-3), (
        f"Benchmark shape ({T},{H},{G}) scales: "
        f"max_diff={(got_s - ref_s).abs().max().item():.4e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mathematical invariants
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES[:3],
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES[:3]])
def test_invariant_output_shape(T, H, G, seed):
    """Output shapes must match: x_q ∈ R^{T×H}, x_s ∈ R^{T×G}."""
    got_q, got_s = kernel_outputs(T, H, G, seed)
    assert got_q.shape == (T, H),      f"x_q shape wrong: {got_q.shape}"
    assert got_s.shape == (T, H // G), f"x_s shape wrong: {got_s.shape}"


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES,
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES])
def test_invariant_quantized_range(T, H, G, seed):
    """x_q must satisfy: all values ∈ [-448.0, 448.0].
    This is the ℓ∞-ball clamp to FP8 E4M3 range (Definition 2 in the paper).
    """
    got_q, _ = kernel_outputs(T, H, G, seed)
    assert got_q.max().item() <=  FP8_MAX + 1e-5, \
        f"x_q exceeds FP8_MAX: max={got_q.max().item()}"
    assert got_q.min().item() >= FP8_MIN - 1e-5, \
        f"x_q below FP8_MIN: min={got_q.min().item()}"


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES,
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES])
def test_invariant_scales_positive(T, H, G, seed):
    """Scale factors must be strictly positive: σ_{τ,γ} > 0 ∀ τ, γ.
    Follows from: σ = max(‖v‖_∞, ε) / α with ε, α > 0.
    """
    _, got_s = kernel_outputs(T, H, G, seed)
    assert got_s.min().item() > 0.0, \
        f"Non-positive scale: min={got_s.min().item()}"


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES,
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES])
def test_invariant_absmax_saturates(T, H, G, seed):
    """For each group, max|x_q| in the group ≈ 448.0 (when the input
    is not all-zeros). Algebraically: σ = ‖v‖_∞/α ⟹ ‖v/σ‖_∞ = α.
    We allow a small tolerance from floating-point rounding.
    """
    x, x_q, x_s = make_input(T, H, G, seed)
    got_q, got_s = custom_kernel((x, x_q, x_s))

    G_count = H // G
    q_grouped = got_q.reshape(T, G_count, G)
    per_group_max = q_grouped.abs().amax(dim=-1)  # [T, G_count]

    # Each group's max absolute quantized value should equal FP8_MAX
    # (unless the input group was all-zeros, which is practically never).
    tol = 1e-2
    non_trivial = per_group_max > 1.0   # skip near-zero groups
    if non_trivial.any():
        max_vals = per_group_max[non_trivial]
        assert (max_vals - FP8_MAX).abs().max().item() < tol, (
            f"Absmax saturation violated: max|x_q| per group should ≈ {FP8_MAX}. "
            f"max_deviation={(max_vals - FP8_MAX).abs().max().item():.4e}"
        )


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES[:3],
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES[:3]])
def test_invariant_dequant_recovers_input(T, H, G, seed):
    """Dequantization x̂ = x_q * σ should reconstruct x within 1/α relative error.
    Algebraically: ‖x̂ - x‖_∞ / ‖x‖_∞ ≤ 1/α + ε for the group ℓ∞ norm.
    """
    x, x_q, x_s = make_input(T, H, G, seed)
    got_q, got_s = custom_kernel((x, x_q, x_s))

    G_count = H // G
    # Dequantize: multiply quantized values by scale
    got_s_expanded = got_s.unsqueeze(-1).expand(T, G_count, G).reshape(T, H)
    x_recon = got_q * got_s_expanded

    # Relative error per element (guard against near-zero x)
    abs_x = x.abs().clamp(min=1e-6)
    rel_err = ((x_recon - x).abs() / abs_x).max().item()
    # Theoretical max relative error: 1/448 ≈ 0.00223 (one FP8 step)
    # Allow 10× for quantization rounding
    assert rel_err < 0.03, \
        f"Dequant relative error too large: {rel_err:.4e} (expected < 0.03)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Output tensor properties
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES[:2],
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES[:2]])
def test_output_dtype_is_float32(T, H, G, seed):
    """Both outputs must be float32 (per task.yml spec)."""
    got_q, got_s = kernel_outputs(T, H, G, seed)
    assert got_q.dtype == torch.float32, f"x_q dtype: {got_q.dtype}"
    assert got_s.dtype == torch.float32, f"x_s dtype: {got_s.dtype}"


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES[:2],
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES[:2]])
def test_output_is_on_cuda(T, H, G, seed):
    """Outputs must reside on CUDA device."""
    got_q, got_s = kernel_outputs(T, H, G, seed)
    assert got_q.is_cuda, "x_q not on CUDA"
    assert got_s.is_cuda, "x_s not on CUDA"


@pytest.mark.parametrize("T,H,G,seed", HACKATHON_TEST_SHAPES[:2],
    ids=[f"T{t}_H{h}_G{g}" for t,h,g,_ in HACKATHON_TEST_SHAPES[:2]])
def test_no_nans_or_infs(T, H, G, seed):
    """Output must be finite: no NaN or Inf values."""
    got_q, got_s = kernel_outputs(T, H, G, seed)
    assert torch.isfinite(got_q).all(), "NaN/Inf in x_q"
    assert torch.isfinite(got_s).all(), "NaN/Inf in x_s"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_all_zeros_input():
    """All-zero input: scales should be eps/FP8_MAX, quantized values = 0."""
    T, H, G = 4, 512, 128
    x   = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    x_q = torch.empty_like(x)
    x_s = torch.empty(T, H // G, dtype=torch.float32, device="cuda")
    got_q, got_s = custom_kernel((x, x_q, x_s))

    assert torch.allclose(got_q, torch.zeros_like(got_q)), \
        "All-zero input should produce all-zero x_q"
    expected_scale = FP8_EPS / FP8_MAX
    assert torch.allclose(got_s, torch.full_like(got_s, expected_scale), rtol=1e-4), \
        f"All-zero input scale should be eps/FP8_MAX={expected_scale:.2e}"


def test_single_nonzero_element_per_group():
    """Single nonzero per group: that element should saturate to ±448."""
    T, H, G = 1, 256, 64
    x = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    # Set first element of each group to 1.0
    x[0, ::G] = 1.0
    x_q = torch.empty_like(x)
    x_s = torch.empty(T, H // G, dtype=torch.float32, device="cuda")
    got_q, got_s = custom_kernel((x, x_q, x_s))

    # The single nonzero element should saturate to FP8_MAX after quantization
    for g_idx in range(H // G):
        val = got_q[0, g_idx * G].item()
        assert abs(val - FP8_MAX) < 1.0, \
            f"Group {g_idx}: first element should ≈ {FP8_MAX}, got {val}"


def test_deterministic_across_runs():
    """Same input → same output on repeated calls (no random state)."""
    T, H, G, seed = 16, 1024, 64, 1234
    x, x_q1, x_s1 = make_input(T, H, G, seed)
    x, x_q2, x_s2 = make_input(T, H, G, seed)   # same seed → same x

    q1, s1 = custom_kernel((x.clone(), x_q1, x_s1))
    q2, s2 = custom_kernel((x.clone(), x_q2, x_s2))

    assert torch.equal(q1, q2), "Non-deterministic x_q across runs"
    assert torch.equal(s1, s2), "Non-deterministic x_s across runs"


def test_large_values_do_not_overflow():
    """Values near float32 max: should clamp cleanly, not produce NaN/Inf."""
    T, H, G = 1, 256, 64
    x = torch.full((T, H), 1e30, dtype=torch.float32, device="cuda")
    x_q = torch.empty_like(x)
    x_s = torch.empty(T, H // G, dtype=torch.float32, device="cuda")
    got_q, got_s = custom_kernel((x, x_q, x_s))

    assert torch.isfinite(got_q).all(), "Overflow: NaN/Inf in x_q for large input"
    assert torch.isfinite(got_s).all(), "Overflow: NaN/Inf in x_s for large input"
    assert got_q.abs().max().item() <= FP8_MAX + 1e-3, "Large input: x_q exceeds FP8_MAX"
