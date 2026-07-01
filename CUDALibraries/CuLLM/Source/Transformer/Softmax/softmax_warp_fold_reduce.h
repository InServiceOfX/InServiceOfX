#ifndef TRANSFORMER_SOFTMAX_SOFTMAX_WARP_FOLD_REDUCE_H
#define TRANSFORMER_SOFTMAX_SOFTMAX_WARP_FOLD_REDUCE_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"

namespace Transformer
{
namespace Softmax
{

//------------------------------------------------------------------------------
/// Maps the I/O type T to the accumulation type for SafeSoftmaxAccumulator.
///
/// Default (float, __half, bfloat16, ...): accumulate in float.
///   sum = Σ exp(x_i - max_value) is bounded by C (the row length) since every
///   term exp(x_i - max_value) ∈ [0,1]. For any realistic sequence length C,
///   sum << FLT_MAX, so overflow is not a concern.
///
/// double: accumulate in double, preserving the precision T = double was chosen
///   for. Using float here would corrupt max_value and therefore every
///   x_i - max_value subtraction.
//------------------------------------------------------------------------------
template <typename T>
struct AccumulationType
{
  using type = float;
};

template <>
struct AccumulationType<double>
{
  using type = double;
};

template <typename T>
using accumulation_type_t = typename AccumulationType<T>::type;

//------------------------------------------------------------------------------
/// Element of the safe-softmax accumulation monoid (S, ⊕, e) where
///   S = R̄ × R≥0,   e = (-∞, 0),   ⊕ = the merge operation below.
///
/// An element (max_value, sum) encodes the statistics of a subsequence x:
///   max_value = m(x) = max_i x_i           — running maximum
///   sum       = ℓ(x) = Σ_i exp(x_i - m)   — sum of shifted exponentials
///
/// Shifting by m keeps every term exp(x_i - m) ∈ (0, 1], preventing
/// overflow. The final softmax output is exp(x_i - m) / ℓ, which equals
/// exp(x_i) / Σ_j exp(x_j) algebraically but is numerically stable.
///
/// AccT is the accumulation precision, selected by AccumulationType<T>.
//------------------------------------------------------------------------------
template <typename AccT>
struct SafeSoftmaxAccumulator
{
  AccT max_value; // m(x) = max_i x_i
  AccT sum;       // ℓ(x) = Σ_i exp(x_i - m(x))
};

//------------------------------------------------------------------------------
/// Merges two accumulators representing disjoint subsequences A and B
/// into a single accumulator for A ∪ B.
///
/// Given a = (m_A, ℓ_A) and b = (m_B, ℓ_B):
///   m_{A∪B} = max(m_A, m_B)
///   ℓ_{A∪B} = ℓ_A · exp(m_A − m_{A∪B}) + ℓ_B · exp(m_B − m_{A∪B})
///
/// The exp factors rescale each partial sum to the common maximum m_{A∪B}
/// before adding, keeping the result numerically stable.
///
/// ⊕ is associative and commutative because it computes the statistics of a
/// set union, and set union is both. This is what makes cg::reduce valid in
/// softmax_warp_fold_reduce: the warp tree can combine partial accumulators
/// in any order and still arrive at the correct global (m, ℓ).
///
/// Identity element: e = (−∞, 0) represents the empty subsequence —
/// max(m_A, −∞) = m_A and ℓ_A + 0 · exp(…) = ℓ_A.
//------------------------------------------------------------------------------
template <typename AccT>
// forceinline helps avoid function call overhead
__device__ __forceinline__ SafeSoftmaxAccumulator<AccT> merge(
  const SafeSoftmaxAccumulator<AccT> a,
  const SafeSoftmaxAccumulator<AccT> b)
{
  const bool a_is_larger {a.max_value > b.max_value};
  const SafeSoftmaxAccumulator<AccT> larger {a_is_larger ? a : b};
  const SafeSoftmaxAccumulator<AccT> smaller {a_is_larger ? b : a};
  // Guard: if smaller holds the identity {-inf, 0}, skip the exponent entirely.
  // Without this, two identity accumulators produce -inf - (-inf) = NaN in the
  // exponent, and 0 * NaN = NaN by IEEE 754 — corrupting the reduction.
  // This arises in the Level 2 warp reduction when C < 32 and some threads own
  // no elements (their partial stays at identity throughout Level 1).
  if (smaller.sum == AccT{0})
  {
    return larger;
  }
  return SafeSoftmaxAccumulator<AccT>{
    larger.max_value,
    larger.sum + smaller.sum *
      Numerics::MathFunctions::get_exponential<AccT>(
        smaller.max_value - larger.max_value)
  };
}

//------------------------------------------------------------------------------
/// Two-level fold of the safe-softmax statistics over one row of length C,
/// using one warp (32 threads) per row.
///
/// Level 1 — sequential fold within each thread (thread coarsening):
///   The warp has 32 threads but the row has C elements. Thread t owns
///   indices t, t+32, t+64, ... and folds each into a local partial
///   Accumulator via merge(). After the loop, each thread holds the
///   safe-softmax statistics (m, ℓ) for its C/32 elements.
///
/// Level 2 — parallel fold across all 32 threads (warp tree reduction):
///   cg::reduce combines the 32 partial accumulators into one total
///   in O(log 32) = 5 warp shuffle steps. Valid because merge() is
///   commutative and associative (it computes statistics of a set union),
///   so the tree can combine partial results in any order.
///
/// Then a normalization pass writes softmax(x)_i = exp(x_i - m) / ℓ.
///
/// T is the I/O type (float, double, __half).
/// AccT = accumulation_type_t<T>: float for all types except double.
/// Specialize for T = __half if __half2 vectorized loads are desired.
//------------------------------------------------------------------------------
template <typename T>
__global__ void softmax_warp_fold_reduce(
  T* output,
  const T* input,
  const int N,
  const int C)
{
  using AccT = accumulation_type_t<T>;
  using Accumulator = SafeSoftmaxAccumulator<AccT>;

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  constexpr int WARP_SIZE {32};
  cg::thread_block_tile<WARP_SIZE> warp {cg::tiled_partition<WARP_SIZE>(block)};

  const int row_index {static_cast<int>(
    blockIdx.x * warp.meta_group_size() + warp.meta_group_rank())};
  if (row_index >= N)
  {
    return;
  }

  const T* x {input + row_index * C};

  // Level 1: sequential fold within each thread (thread coarsening).
  // The warp has 32 threads but the row has C elements. Thread t owns
  // indices t, t+32, t+64, ... and folds each into its local partial
  // accumulator via merge(). After this loop every thread holds one
  // Accumulator representing the safe-softmax statistics (m, ℓ) for its
  // C/32 elements. Identity element of (S, ⊕): (-∞, 0).
  Accumulator partial {
    static_cast<AccT>(-Numerics::Constants::get_infinity<AccT>()), AccT{0}};
  for (int i {static_cast<int>(warp.thread_rank())}; i < C;
    i += static_cast<int>(warp.size()))
  {
    partial = merge(partial, Accumulator{static_cast<AccT>(x[i]), AccT{1}});
  }

  // Level 2: parallel fold across all 32 threads (warp tree reduction).
  // cg::reduce combines the 32 partial accumulators into one total using
  // warp shuffle instructions in O(log 32) = 5 steps. Valid because merge
  // is commutative and associative, so the tree can combine partial results in
  // any order.
  const Accumulator total {cg::reduce(warp, partial, merge<AccT>)};

  // Normalization pass: softmax(x)_i = exp(x_i - m) / ℓ .
  for (int i {static_cast<int>(warp.thread_rank())}; i < C;
    i += static_cast<int>(warp.size()))
  {
    output[row_index * C + i] = static_cast<T>(
      Numerics::MathFunctions::get_exponential<AccT>(
        static_cast<AccT>(x[i]) - total.max_value) / total.sum);
  }
}

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_SOFTMAX_WARP_FOLD_REDUCE_H
