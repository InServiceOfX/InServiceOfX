#ifndef LLM_ATTENTION_FORWARD_SOFTMAX_H
#define LLM_ATTENTION_FORWARD_SOFTMAX_H

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "ParallelProcessing/warp_reductions.h"

namespace LLM
{
namespace AttentionForward
{

template <typename FPType>
__global__ void softmax_forward_kernel4(
  FPType* out,
  const FPType* input,
  const int N,
  const int C)
{
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // same as kernel3, but can handle any block size (multiple of 32)
  // each row of C elements is handled by block_size threads
  // furthermore, each block_size threads get executed in warps of 32 threads

  // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
  // shared memory is used for inter-warp reduction
  extern __shared__ FPType shared[];
  const size_t idx {blockIdx.x};
  const size_t tid {threadIdx.x};
  const size_t warpId {threadIdx.x / 32}; // warp index within a block
  const size_t laneId {threadIdx.x % 32}; // thread index within a warp

  // the number of warps per block. recall that blockDim.x is block_size
  const size_t warpsPerBlock {blockDim.x / 32};

  // shared[] must be allocated to have 2 * warpsPerBlock elements
  // first half for max values, the second half for sum values
  FPType* maxvals {shared};
  FPType* sumvals {&shared[warpsPerBlock]};

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  const FPType* x {input + idx * C};

  // first, thread coarsening by directly accessing global memory in series
  FPType maxval {-Numerics::Constants::get_infinity<FPType>()};
  for (size_t i {tid}; i < C; i += blockDim.x)
  {
    maxval = Numerics::MathFunctions::get_max<FPType>(maxval, x[i]);
  }

  // now within-warp reductions for maxval
  maxval = ParallelProcessing::warp_reduce_max(maxval);

  // the 0th thread of each warp writes the maxval of that warp to shared memory
  if (laneId == 0)
  {
    maxvals[warpId] = maxval;
  }
  __syncthreads();

  // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
  if (tid == 0)
  {
    FPType val {maxvals[tid]};
    for (int i {1}; i < warpsPerBlock; ++i)
    {
      val = Numerics::MathFunctions::get_max<FPType>(val, maxvals[i]);
    }
    // store the final max in the first position
    maxvals[0] = val;
  }
  __syncthreads();
  // broadcast the max to all threads
  FPType offset {maxvals[0]};

  // compute expf and write the result to global memory
  for (int i {tid}; i < C; i += blockDim.x)
  {
    // subtract max for numerical stability
    out[idx * C + i] = Numerics::MathFunctions::get_exponential<FPType>(
      x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  FPType sumval {0.0f};
  for (int i {tid}; i < C; i += blockDim.x)
  {
    sumval += x[i];
  }
  // within-warp reduction for sumval
  //sumval = warpReduceSum(sumval);

  // write sumval to shared memory
  if (laneId == 0)
  {
    sumvals[warpId] = sumval;
  }
  __syncthreads();

  // inter-thread reduction of sum
  if (tid == 0)
  {
    FPType val {sumvals[tid]};
    for (int i {1}; i < warpsPerBlock; ++i)
    {
      val += sumvals[i];
    }
    sumvals[0] = val;
  }
  __syncthreads();
  // broadcast the sum to all threads
  FPType sum {sumvals[0]};

  // divide the whole row by the sum
  for (int i {tid}; i < C; i += blockDim.x)
  {
    out[idx * C + i] = x[i] / sum;
  }
}

} // namespace AttentionForward
} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_SOFTMAX_H