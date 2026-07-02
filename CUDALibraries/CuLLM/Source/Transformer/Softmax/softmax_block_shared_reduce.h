#ifndef TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_SHARED_REDUCE_H
#define TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_SHARED_REDUCE_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"
#include "Utilities/Memory/streaming_store.h"

namespace Transformer
{
namespace Softmax
{

//------------------------------------------------------------------------------
/// Two-pass block-wide softmax over one row of length C, using one block
/// (one or more warps) per row.
///
/// Compared to softmax_warp_fold_reduce (one warp = 32 threads per row),
/// this kernel assigns a full block of blockDim.x threads to each row.
/// This covers larger C by distributing work across more threads and
/// coordinates across warps via shared memory.
///
/// Algorithm — three passes over row data:
///
///   Pass 1 — find m = max_i x_i:
///     Each thread folds its strided elements (tid, tid+blockDim.x, ...) into
///     a local maximum. cg::reduce collapses the 32 threads of each warp to a
///     warp maximum. Warp lane-0 writes to shared memory; thread 0 reduces
///     across warps and stores the block maximum in shared[0]. block.sync()
///     broadcasts it to all threads.
///
///   Pass 2 — write exp(x_i − m) to output:
///     Standard write-back stores are used deliberately so the intermediate
///     exp values remain in L2 for Pass 3 to read. Using streaming_store here
///     would mark those lines evict-first and turn the Pass 3 reads into L2
///     misses.
///
///   Pass 3 — find ℓ = Σ_i exp(x_i − m), normalize, write output:
///     Reads the exp values written in Pass 2. Same two-level reduction as
///     Pass 1 but with cg::plus<AccT>{} instead of get_max. After the block
///     sum is broadcast, each thread writes softmax(x)_i = y[i] / ℓ via
///     streaming_store — these normalized values are the final output and will
///     not be reused by this kernel, so keeping them in cache would only
///     displace data other warps need.
///
/// Shared memory layout: one AccT slot per warp, reused for both passes.
///   Size = (blockDim.x / 32) * sizeof(AccT).
///
/// Launch: <<<N, block_size, (block_size / 32) * sizeof(AccT)>>>
///   block_size must be a multiple of 32.
///
/// T   — I/O type (float, double, __half).
/// AccT = accumulation_type_t<T>: float for all types except double.
//------------------------------------------------------------------------------
template <typename T>
__global__ void softmax_block_shared_reduce(
  T* output,
  const T* input,
  const int N,
  const int C)
{
  using AccT = accumulation_type_t<T>;

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  cg::thread_block_tile<32> warp {cg::tiled_partition<32>(block)};

  const int row_index {static_cast<int>(blockIdx.x)};
  if (row_index >= N)
  {
    return;
  }

  const int tid {static_cast<int>(threadIdx.x)};
  const int warps_per_block {static_cast<int>(warp.meta_group_size())};

  // Shared memory: one AccT per warp for inter-warp reduction.
  // Reused for both the max pass and the sum pass.
  // Use unsigned char + reinterpret_cast to avoid linker conflicts when
  // multiple template instantiations (float, double) are in the same TU.
  extern __shared__ unsigned char smem[];
  AccT* shared {reinterpret_cast<AccT*>(smem)};

  const T* x {input + row_index * C};

  //----------------------------------------------------------------------------
  // Pass 1: find m = max_i x_i.
  //
  // Thread tid covers x[tid], x[tid + blockDim.x], x[tid + 2*blockDim.x], ...
  // folding into thread_max. Threads with no elements keep -infinity, the
  // identity for max, and contribute nothing to the reduction.
  //----------------------------------------------------------------------------
  AccT thread_max {-Numerics::Constants::get_infinity<AccT>()};
  for (int i {tid}; i < C; i += static_cast<int>(blockDim.x))
  {
    thread_max = Numerics::MathFunctions::get_max<AccT>(
      thread_max, static_cast<AccT>(x[i]));
  }
  // Intra-warp max: cg::reduce with get_max, O(log 32) warp shuffles.
  const AccT warp_max {cg::reduce(warp, thread_max,
    Numerics::MathFunctions::get_max<AccT>)};
  // Lane-0 of each warp writes its warp maximum to shared memory.
  if (warp.thread_rank() == 0)
  {
    shared[warp.meta_group_rank()] = warp_max;
  }
  block.sync();
  // Thread 0 reduces across warps and broadcasts via shared[0].
  if (tid == 0)
  {
    AccT block_max {shared[0]};
    for (int w {1}; w < warps_per_block; ++w)
    {
      block_max = Numerics::MathFunctions::get_max<AccT>(block_max, shared[w]);
    }
    shared[0] = block_max;
  }
  block.sync();
  const AccT max_value {shared[0]};

  //----------------------------------------------------------------------------
  // Pass 2: write exp(x_i − max_value) to output.
  //
  // Standard stores (not streaming) so Pass 3 can read these from L2 cache.
  //----------------------------------------------------------------------------
  for (int i {tid}; i < C; i += static_cast<int>(blockDim.x))
  {
    output[row_index * C + i] = static_cast<T>(
      Numerics::MathFunctions::get_exponential<AccT>(
        static_cast<AccT>(x[i]) - max_value));
  }
  block.sync();

  //----------------------------------------------------------------------------
  // Pass 3: sum ℓ = Σ_i exp(x_i − m), then write normalized output.
  //
  // Same two-level reduction as Pass 1 but summing with cg::plus<AccT>{}.
  // shared[] is reused — the max pass result is no longer needed.
  //----------------------------------------------------------------------------
  const T* y {output + row_index * C};
  AccT thread_sum {AccT{0}};
  for (int i {tid}; i < C; i += static_cast<int>(blockDim.x))
  {
    thread_sum += static_cast<AccT>(y[i]);
  }
  // Intra-warp sum.
  const AccT warp_sum {cg::reduce(warp, thread_sum, cg::plus<AccT>{})};
  if (warp.thread_rank() == 0)
  {
    shared[warp.meta_group_rank()] = warp_sum;
  }
  block.sync();
  if (tid == 0)
  {
    AccT block_sum {shared[0]};
    for (int w {1}; w < warps_per_block; ++w)
    {
      block_sum += shared[w];
    }
    shared[0] = block_sum;
  }
  block.sync();
  const AccT sum {shared[0]};

  // Normalization: softmax(x)_i = exp(x_i − m) / ℓ = y[i] / sum.
  // streaming_store: final output, not reused by this kernel.
  for (int i {tid}; i < C; i += static_cast<int>(blockDim.x))
  {
    Utilities::Memory::streaming_store(
      output + row_index * C + i,
      static_cast<T>(static_cast<AccT>(y[i]) / sum));
  }
}

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_SHARED_REDUCE_H
