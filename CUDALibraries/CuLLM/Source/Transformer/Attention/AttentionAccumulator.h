#ifndef TRANSFORMER_ATTENTION_ATTENTION_ACCUMULATOR_H
#define TRANSFORMER_ATTENTION_ATTENTION_ACCUMULATOR_H

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// Attention output accumulator (§20 Definition 20.1 of FlashAttention.tex).
///
/// For a fixed query row q ∈ R^{d_k} and a subset A ⊆ {1,...,n} of key-value
/// positions, the triple
///
///   α(A) = (m, ℓ, õ)  ∈  R̄ × R≥0 × R^{d_v}
///
/// encodes:
///   m   = m(s_A) = max_{j∈A} s_j                     — running maximum score
///   ℓ   = ℓ(s_A) = Σ_{j∈A} exp(s_j − m)              — sum of shifted exps
///   õ   = Σ_{j∈A} exp(s_j − m) · V_{j,·} ∈ R^{d_v}  — unnormalized partial output
///
/// where s_j = q · K_{j,·}^⊤ / √{d_k} is the scaled attention score and
/// V_{j,·} is the j-th row of the value matrix.
///
/// This extends SafeSoftmaxAccumulator (which tracks only the pair (m, ℓ)) by
/// adding the partial output vector õ. The normalized output for subset A is
///   o(A) = õ(A) / ℓ(A)  ∈ R^{d_v}.
///
/// Identity element: (-∞, 0, 0_{d_v}) — both ℓ and all d_v components of õ
/// are zero, so merging with it leaves the other accumulator unchanged.
///
/// Unlike SafeSoftmaxAccumulator, this type is NOT suitable for cg::reduce
/// (warp shuffle operates on ≤32-bit values; the output[kHeadDim] array cannot
/// pass through CUDA warp shuffles). It is designed for sequential accumulation:
/// the FlashAttention inner loop merges one K/V tile accumulator at a time into
/// the running total without ever writing the N×N score matrix to HBM.
///
/// kHeadDim must be a compile-time constant so output[] lives in registers.
/// AccT — accumulation precision (float or double).
//------------------------------------------------------------------------------
template <typename AccT, int kHeadDim>
struct AttentionAccumulator
{
  // m(s_A) = max_{j∈A} s_j
  AccT max_value;
  // ℓ(s_A) = Σ_{j∈A} exp(s_j − m(s_A))
  AccT sum;
  // õ(A) = Σ_{j∈A} exp(s_j − m(s_A)) · V_{j,·}
  AccT output[kHeadDim];
};

//------------------------------------------------------------------------------
/// Returns the identity element (-∞, 0, 0_{d_v}) for the merge monoid.
//------------------------------------------------------------------------------
template <typename AccT, int kHeadDim>
__device__ __forceinline__ AttentionAccumulator<AccT, kHeadDim> attention_identity()
{
  AttentionAccumulator<AccT, kHeadDim> id;
  id.max_value = -Numerics::Constants::get_infinity<AccT>();
  id.sum = AccT{0};
  #pragma unroll
  for (int d {0}; d < kHeadDim; ++d)
  {
    id.output[d] = AccT{0};
  }
  return id;
}

//------------------------------------------------------------------------------
/// Merges two attention accumulators for disjoint subsets A and B
/// into one accumulator for A ∪ B (§20 Proposition 20.2 of FlashAttention.tex).
///
/// Given α(A) = (m_A, ℓ_A, õ_A) and α(B) = (m_B, ℓ_B, õ_B):
///   m_{A∪B}  = max(m_A, m_B)
///   ℓ_{A∪B}  = exp(m_A − m) · ℓ_A   + exp(m_B − m) · ℓ_B
///   õ_{A∪B}  = exp(m_A − m) · õ_A   + exp(m_B − m) · õ_B
///
/// where m = m_{A∪B}. The exp factors rescale both ℓ and each component of õ
/// to the common maximum m before adding, keeping the result numerically stable.
///
/// The resulting triple (m, ℓ_{A∪B}, õ_{A∪B}) also forms a commutative monoid
/// (§20 Remark: "Second commutative monoid"). The FlashAttention algorithm is
/// the sequential left fold ⊕_{t=1}^{T} α(A_t) over K/V tiles, which equals
/// α({1,...,n}) — the accumulator for the full sequence — without ever
/// materializing the N×N attention score matrix in HBM.
//------------------------------------------------------------------------------
template <typename AccT, int kHeadDim>
__device__ __forceinline__ AttentionAccumulator<AccT, kHeadDim> merge(
  const AttentionAccumulator<AccT, kHeadDim> a,
  const AttentionAccumulator<AccT, kHeadDim> b)
{
  // Guard: a zero sum means the accumulator is the identity element (-∞, 0, 0).
  // Without this, exp(-∞ − (-∞)) = exp(NaN) propagates through ℓ and õ.
  if (a.sum == AccT{0})
  {
    return b;
  }
  if (b.sum == AccT{0})
  {
    return a;
  }

  const AccT m {Numerics::MathFunctions::get_max<AccT>(a.max_value, b.max_value)};
  const AccT scale_a {Numerics::MathFunctions::get_exponential<AccT>(a.max_value - m)};
  const AccT scale_b {Numerics::MathFunctions::get_exponential<AccT>(b.max_value - m)};

  AttentionAccumulator<AccT, kHeadDim> result;
  result.max_value = m;
  result.sum = scale_a * a.sum + scale_b * b.sum;
  #pragma unroll
  for (int d {0}; d < kHeadDim; ++d)
  {
    result.output[d] = scale_a * a.output[d] + scale_b * b.output[d];
  }
  return result;
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_ATTENTION_ACCUMULATOR_H
