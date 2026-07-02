#ifndef TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_UNROLLED_FUSED_H
#define TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_UNROLLED_FUSED_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"
#include "Utilities/Memory/streaming_load.h"
#include "Utilities/Memory/streaming_store.h"

namespace Transformer
{
namespace Softmax
{

//------------------------------------------------------------------------------
/// Block-wide softmax over one row of length C, optimised for large C via
/// register-array unrolling — one block per row.
///
/// Compared to softmax_block_shared_reduce (same two-level reduction but no
/// unrolling), this kernel issues kUnrollFactor global loads per thread per
/// outer-loop iteration in two separate loops: one to fill a register array,
/// one to process it. This decoupling lets the compiler overlap the load
/// latencies with each other (memory-level parallelism), which is the dominant
/// win for large C where arithmetic is cheap relative to DRAM bandwidth.
///
/// Algorithm — three passes over row data:
///
///   Pass 1 — find m = max_i x_i:
///     Unconditional reads with clamped index (min(C-1, i + u*blockDim.x)) so
///     all kUnrollFactor loads can be issued without branch divergence. OOB
///     slots read x[C-1] redundantly; max is idempotent so the duplicates are
///     harmless. Uses standard (non-streaming) loads to warm x[] in L2 for
///     Pass 2. Two-level reduction: cg::reduce within each warp, then thread 0
///     reduces warp maxima in max_shared[].
///
///   Pass 2 — fused exp + sum (streaming inputs):
///     Loads kUnrollFactor values from x[] via streaming_load (__ldcs), placing
///     them in a register array. Separate #pragma unroll loops for loading and
///     processing ensure the compiler issues all loads before any arithmetic.
///     Conditional writes (if col < C) compute exp(x_i - m) and write to
///     output with standard stores — kept in L2 so Pass 3 reads are cache hits.
///     Same-iteration sum accumulation avoids a separate pass over output[].
///
///   Pass 3 — normalize:
///     Reads the exp values from output[] into a register array (unconditional,
///     clamped index), then writes softmax(x)_i = exp(x_i - m) / ℓ via
///     streaming_store — these normalized values are the final output and will
///     not be read back by this kernel.
///
/// The block.sync() that separates Pass 2 from the sum reduction also acts as
/// a global-memory fence, guaranteeing that Pass 2's output[] writes are
/// visible to all threads before Pass 3 reads them.
///
/// Shared memory layout: two halves of warps_per_block AccT slots each.
///   max_shared = smem[0 .. warps_per_block-1]
///   sum_shared = smem[warps_per_block .. 2*warps_per_block-1]
///   Total size = 2 * (blockDim.x / 32) * sizeof(AccT).
///
/// Launch: <<<N, block_size, 2 * (block_size / 32) * sizeof(AccT)>>>
///   block_size must be a multiple of 32.
///   kUnrollFactor = 8 matches the default in llm.c softmax_forward_kernel7.
///
/// T            — I/O type (float, double, __half).
/// kUnrollFactor — register-array depth; must be a compile-time constant.
/// AccT = accumulation_type_t<T>: float for all types except double.
//------------------------------------------------------------------------------
template <typename T, int kUnrollFactor = 8>
__global__ void softmax_block_unrolled_fused(
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
  const int block_width {static_cast<int>(blockDim.x)};
  const int warps_per_block {static_cast<int>(warp.meta_group_size())};

  // Shared memory: two halves of warps_per_block AccT slots each.
  // First half: per-warp max values (Pass 1 inter-warp reduction).
  // Second half: per-warp sum values (Pass 2 inter-warp reduction).
  extern __shared__ unsigned char smem[];
  AccT* max_shared {reinterpret_cast<AccT*>(smem)};
  AccT* sum_shared {reinterpret_cast<AccT*>(smem) + warps_per_block};

  const T* x {input + row_index * C};

  //----------------------------------------------------------------------------
  // Pass 1: find m = max_i x_i.
  //
  // Unconditional reads with clamped index so the compiler can issue
  // kUnrollFactor loads in parallel. OOB slots read x[C-1]; max is
  // idempotent, so duplicates do not change the result.
  // Standard reads (not streaming_load) to warm x[] in L2 for Pass 2.
  //----------------------------------------------------------------------------
  AccT thread_max {-Numerics::Constants::get_infinity<AccT>()};
  for (int i {tid}; i < C; i += block_width * kUnrollFactor)
  {
    #pragma unroll
    for (int u {0}; u < kUnrollFactor; ++u)
    {
      thread_max = Numerics::MathFunctions::get_max<AccT>(
        thread_max,
        static_cast<AccT>(x[min(C - 1, i + u * block_width)]));
    }
  }
  const AccT warp_max {cg::reduce(warp, thread_max,
    Numerics::MathFunctions::get_max<AccT>)};
  if (warp.thread_rank() == 0)
  {
    max_shared[warp.meta_group_rank()] = warp_max;
  }
  block.sync();
  if (tid == 0)
  {
    AccT block_max {max_shared[0]};
    for (int w {1}; w < warps_per_block; ++w)
    {
      block_max = Numerics::MathFunctions::get_max<AccT>(block_max, max_shared[w]);
    }
    max_shared[0] = block_max;
  }
  block.sync();
  const AccT max_value {max_shared[0]};

  //----------------------------------------------------------------------------
  // Pass 2 (fused): load x via streaming_load, compute exp(x_i - m),
  // write to output (standard stores so L2 stays warm for Pass 3),
  // accumulate sum into thread_sum.
  //
  // Two separate #pragma unroll loops: first fills the register array with
  // kUnrollFactor streaming loads issued in parallel; second processes each
  // element and conditionally writes. The separation is the key compiler hint
  // that keeps the loads independent for memory-level parallelism.
  //----------------------------------------------------------------------------
  AccT thread_sum {AccT{0}};
  for (int i {tid}; i < C; i += block_width * kUnrollFactor)
  {
    AccT reg_array[kUnrollFactor];
    #pragma unroll
    for (int u {0}; u < kUnrollFactor; ++u)
    {
      reg_array[u] = static_cast<AccT>(
        Utilities::Memory::streaming_load<T>(
          &x[min(C - 1, i + u * block_width)]));
    }
    #pragma unroll
    for (int u {0}; u < kUnrollFactor; ++u)
    {
      if (i + u * block_width < C)
      {
        const AccT exp_val {Numerics::MathFunctions::get_exponential<AccT>(
          reg_array[u] - max_value)};
        output[row_index * C + i + u * block_width] = static_cast<T>(exp_val);
        thread_sum += exp_val;
      }
    }
  }

  // Sum reduction: two-level, same pattern as Pass 1 max.
  // The block.sync() here also fences Pass 2 global-memory writes (output[]),
  // so Pass 3 reads of output[] are coherent for all threads.
  const AccT warp_sum {cg::reduce(warp, thread_sum, cg::plus<AccT>{})};
  if (warp.thread_rank() == 0)
  {
    sum_shared[warp.meta_group_rank()] = warp_sum;
  }
  block.sync();
  if (tid == 0)
  {
    AccT block_sum {sum_shared[0]};
    for (int w {1}; w < warps_per_block; ++w)
    {
      block_sum += sum_shared[w];
    }
    sum_shared[0] = block_sum;
  }
  block.sync();
  const AccT sum {sum_shared[0]};

  //----------------------------------------------------------------------------
  // Pass 3: normalize. Read exp values from output[] into a register array
  // (unconditional, clamped), then write softmax(x)_i = exp(x_i-m)/ℓ via
  // streaming_store — the final output will not be read back by this kernel.
  //----------------------------------------------------------------------------
  const T* y {output + row_index * C};
  for (int i {tid}; i < C; i += block_width * kUnrollFactor)
  {
    AccT reg_array[kUnrollFactor];
    #pragma unroll
    for (int u {0}; u < kUnrollFactor; ++u)
    {
      reg_array[u] = static_cast<AccT>(y[min(C - 1, i + u * block_width)]);
    }
    #pragma unroll
    for (int u {0}; u < kUnrollFactor; ++u)
    {
      if (i + u * block_width < C)
      {
        Utilities::Memory::streaming_store(
          output + row_index * C + i + u * block_width,
          static_cast<T>(reg_array[u] / sum));
      }
    }
  }
}

} // namespace Softmax
} // namespace Transformer

#endif // TRANSFORMER_SOFTMAX_SOFTMAX_BLOCK_UNROLLED_FUSED_H
