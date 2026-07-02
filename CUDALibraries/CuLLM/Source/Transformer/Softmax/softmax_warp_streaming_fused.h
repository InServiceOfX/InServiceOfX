#ifndef TRANSFORMER_SOFTMAX_SOFTMAX_WARP_STREAMING_FUSED_H
#define TRANSFORMER_SOFTMAX_SOFTMAX_WARP_STREAMING_FUSED_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"
#include "Transformer/Softmax/softmax_warp_fold_reduce.h"
#include "Utilities/Memory/streaming_load.h"
#include "Utilities/Memory/streaming_store.h"

namespace Transformer
{
namespace Softmax
{

//------------------------------------------------------------------------------
/// Two-pass warp-wide softmax over one row of length C, using one warp (32
/// threads) per row — same execution mapping and merge-monoid fold as
/// softmax_warp_fold_reduce, plus cache-hint memory operations on the second
/// pass.
///
/// Pass 1 (fold, unchanged from softmax_warp_fold_reduce): each lane reads
/// its strided elements x[tid], x[tid+32], ... with ordinary loads and folds
/// them into a local (m, ℓ) via merge(); cg::reduce combines the 32 lane
/// partials into the row's exact statistics. Ordinary (non-streaming) loads
/// here deliberately warm L1/L2 with x[], since Pass 2 reads the same
/// addresses again — matching the same warming rationale documented in
/// softmax_block_shared_reduce's Pass 1.
///
/// Pass 2 (normalize): recomputes softmax(x)_i = exp(x_i − m)/ℓ directly
/// from x — there is no intermediate exp[] array to write or re-read, so
/// this pass touches x[] exactly once more (never again after) and writes
/// output[] exactly once (never read back by this kernel). Both qualify for
/// the streaming (evict-first) cache hint:
///   streaming_load(&x[i])       — PTX ld.global.cs, x[i] already consumed.
///   streaming_store(&output[i]) — PTX st.global.cs, output[i] is final.
/// This is the same two-touches-of-x/one-write-of-output shape as
/// softmax_warp_fold_reduce; the only change is the cache hints on the
/// second touch of each address.
///
/// T is the I/O type (float, double, __half).
/// AccT = accumulation_type_t<T>: float for all types except double.
//------------------------------------------------------------------------------
template <typename T>
__global__ void softmax_warp_streaming_fused(
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

  // Pass 1: fold (m, ℓ) via the merge monoid. Ordinary loads: x[] is read
  // again in Pass 2, so keep these lines warm in L1/L2 rather than marking
  // them evict-first.
  Accumulator partial {
    static_cast<AccT>(-Numerics::Constants::get_infinity<AccT>()), AccT{0}};
  for (
    int i {static_cast<int>(warp.thread_rank())};
    i < C;
    i += static_cast<int>(warp.size()))
  {
    partial = merge(partial, Accumulator{static_cast<AccT>(x[i]), AccT{1}});
  }
  const Accumulator total {cg::reduce(warp, partial, merge<AccT>)};

  // Pass 2: recompute exp(x_i - m)/ℓ directly from x (no intermediate
  // array). x[i] is never read again after this — streaming_load. The
  // written value is never read back by this kernel — streaming_store.
  for (
    int i {static_cast<int>(warp.thread_rank())};
    i < C;
    i += static_cast<int>(warp.size()))
  {
    const AccT x_i {
      static_cast<AccT>(Utilities::Memory::streaming_load<T>(&x[i]))};
    const T softmax_i {static_cast<T>(
      Numerics::MathFunctions::get_exponential<AccT>(x_i - total.max_value) /
        total.sum)};
    Utilities::Memory::streaming_store(output + row_index * C + i, softmax_i);
  }
}

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_SOFTMAX_WARP_STREAMING_FUSED_H
