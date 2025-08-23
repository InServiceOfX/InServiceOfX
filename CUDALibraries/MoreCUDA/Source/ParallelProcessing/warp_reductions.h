#ifndef PARALLEL_PROCESSING_WARP_REDUCTIONS_H
#define PARALLEL_PROCESSING_WARP_REDUCTIONS_H

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

template <typename T> __device__ T warp_reduce_sum(T value) = delete;

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html?highlight=__shfl_xor_sync#_CPPv415__shfl_xor_syncKjK6__halfKiKi
/// __device__ __half __shfl_xor_sync(
///   const unsigned int mask,
///   const __half var,
///   const int laneMask,
///   const int width = warpSize)
/// Exchange a variable between threads within a warp.
/// Copy from a thread based on bitwise XOR of own thread ID.
/// Calculates source thread ID by performing bitwise XOR of caller's thread ID
/// with laneMask.
//------------------------------------------------------------------------------

template <> __device__ inline float warp_reduce_sum<float>(float value)
{
  for (int offset {16}; offset > 0; offset /= 2)
  {
    // For the given thread, __xhfl_xor_sync() fetches the value from another
    // thread, whose thread ID or lane ID is its own thread ID XOR'ed with the
    // offset.
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

template <> __device__ inline double warp_reduce_sum<double>(double value)
{
  for (int offset {16}; offset > 0; offset /= 2)
  {
    value += __longlong_as_double(__shfl_xor_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

template <> __device__ inline __half warp_reduce_sum<__half>(__half value)
{
  // Convert to float for shuffle
  float val {__half2float(value)};
  for (int offset {16}; offset > 0; offset /= 2)
  {
    // See
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html#_CPPv412__half2floatK6__half
    // __half2float(const __half a) converts half number to float.
    val += __half2float(__shfl_xor_sync(0xFFFFFFFF, value, offset));
  }
  return __float2half(val);  // Convert back to half
}

//------------------------------------------------------------------------------
/// See
/// https://github.com/andrewkchan/yalm/blob/main/src/infer.cpp
/// for warp_reduce_sum implementation for float, using __shfl_down_sync.
/// __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = warpSize)
/// shfl is for shuffle, as these intrinsics perform a shuffle (a rearrangement/
/// exchange) of register data among threads in a warp.
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-description
/// __shfl_down_sync() Copy from a lane with higher ID relative to caller.
/// __shfl_down_sync() calculates a source lane ID by adding delta to the
/// caller's lane ID. The value of var held by resulting lane ID is returned:
/// this has the effect of shifting var down the warp by delta lanes.
//------------------------------------------------------------------------------
template <typename T> __device__ T warp_reduce_sum_with_shuffle_down(T val)
{
  for (int offset {warpSize / 2}; offset > 0; offset /= 2)
  {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

} // namespace ParallelProcessing

#endif // PARALLEL_PROCESSING_WARP_REDUCE_MAX_H