#ifndef PARALLEL_PROCESSING_WARP_REDUCTIONS_H
#define PARALLEL_PROCESSING_WARP_REDUCTIONS_H

#include <cuda_runtime.h>
#include "Numerics/MathFunctions.h"

namespace ParallelProcessing
{

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture
/// 4.1 SIMT Architecture
/// The multiprocessor creates, manages, schedules, and executes threads in
/// groups of 32 parallel threads called warps. Individual threads composing a
/// warp start together at same program address, but they have their own
/// instruction addresss counter and register state and therefore free to branch
/// and execute independently. The term warp originates from weaving, the first
/// parallel thread technology.
///
/// See
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
/// 7.22. Warp Shuffle Functions
/// exchange a variable between threads within a warp.
/// T __shfl_down_sync(unsigned mask, T var, unsigned int delta,
///   int width=warpSize)
/// __shfl_down_sync() calculates source lane ID by adding delta to caller's
/// lane ID. Value of var held by resulting lane ID is returned: this has the
/// effect of shifting var down the warp by delta lanes.
///
/// It appears that warp_reduce_max(..) shall always place the maximum value at
/// the zeroth position of the warp.
//------------------------------------------------------------------------------
template <typename FPType>
__device__ FPType warp_reduce_max(FPType value)
{
  for (int offset {16}; offset > 0; offset /= 2)
  {
    value = Numerics::MathFunctions::get_max<FPType>(
      value,
      // Recall, 0xFFFFFFFF is a 32-bit mask.
      __shfl_down_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

template <typename FPType>
__device__ FPType warp_reduce_max_shift_up(FPType value)
{
  for (int offset {1}; offset < warpSize; offset *= 2)
  {
    value = Numerics::MathFunctions::get_max<FPType>(
      value,
      // Recall, 0xFFFFFFFF is a 32-bit mask.
      __shfl_up_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

template <typename FPType>
__device__ FPType warp_reduce_sum(FPType value)
{
  for (int offset {16}; offset > 0; offset /= 2)
  {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

} // namespace ParallelProcessing

#endif // PARALLEL_PROCESSING_WARP_REDUCE_MAX_H