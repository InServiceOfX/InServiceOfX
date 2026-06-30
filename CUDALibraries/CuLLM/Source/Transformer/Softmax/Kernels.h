#ifndef TRANSFORMER_SOFTMAX_KERNELS_H
#define TRANSFORMER_SOFTMAX_KERNELS_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <type_traits>

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
/// Element of the monoid S = R̄ × R≥0 (Definition 18.3 in FlashAttention.tex).
/// AccT is the accumulation precision, selected by AccumulationType<T>.
//------------------------------------------------------------------------------
template <typename AccT>
struct SafeSoftmaxAccumulator
{
  AccT max_value; // m(x) = max_i x_i
  AccT sum;       // ℓ(x) = Σ_i exp(x_i - m(x))
};

//------------------------------------------------------------------------------
/// The merge operation ⊕ on S (Definition 18.3).
/// Associative and commutative (Proposition 18.4), which is what makes
/// cg::reduce valid in softmax_warp_fold_reduce below.
//------------------------------------------------------------------------------
template <typename AccT>
__device__ __forceinline__ SafeSoftmaxAccumulator<AccT> merge(
  const SafeSoftmaxAccumulator<AccT> a,
  const SafeSoftmaxAccumulator<AccT> b)
{
  const bool a_is_larger = (a.max_value > b.max_value);
  const SafeSoftmaxAccumulator<AccT> larger  = a_is_larger ? a : b;
  const SafeSoftmaxAccumulator<AccT> smaller = a_is_larger ? b : a;
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
///   each thread strides over C/32 elements, accumulating a partial
///   SafeSoftmaxAccumulator<AccT> via merge().
///
/// Level 2 — parallel fold across the warp:
///   cg::reduce combines the 32 partial accumulators using merge() as the
///   binary operator. Valid because merge() is commutative and associative
///   (Proposition 18.4 / Remark 19.2 in FlashAttention.tex).
///
/// Then a normalization pass writes softmax(x)_i = exp(x_i - m) / ℓ.
///
/// T is the I/O type (float, double, __half).
/// AccT = accumulation_type_t<T>: float for all types except double.
/// Specialize for T = __half if __half2 vectorized loads are desired.
//------------------------------------------------------------------------------
template <typename T>
__global__ void softmax_warp_fold_reduce(T* out, const T* inp, int N, int C)
{
  using AccT = accumulation_type_t<T>;
  using Accumulator = SafeSoftmaxAccumulator<AccT>;

  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  const int row = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (row >= N)
  {
    return;
  }

  const T* x = inp + row * C;

  // Level 1: sequential fold — each thread accumulates its strided slice.
  // Identity element of (S, ⊕): (-∞, 0) (Definition 18.3).
  Accumulator partial{static_cast<AccT>(-INFINITY), AccT{0}};
  for (int i = warp.thread_rank(); i < C; i += warp.size())
  {
    partial = merge(partial, Accumulator{static_cast<AccT>(x[i]), AccT{1}});
  }

  // Level 2: parallel fold — warp tree reduction using ⊕.
  const Accumulator total = cg::reduce(warp, partial, merge<AccT>);

  // Normalization pass: softmax(x)_i = exp(x_i - m) / ℓ (Definition 5.2).
  for (int i = warp.thread_rank(); i < C; i += warp.size())
  {
    out[row * C + i] = static_cast<T>(
      Numerics::MathFunctions::get_exponential<AccT>(
        static_cast<AccT>(x[i]) - total.max_value) / total.sum);
  }
}

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_KERNELS_H
