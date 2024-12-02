#ifndef DRAFTS_LLM_ATTENTION_FORWARD_SOFTMAX_H
#define DRAFTS_LLM_ATTENTION_FORWARD_SOFTMAX_H

#include "Numerics/MathFunctions.h"
#include "Numerics/Constants/get_infinity.h"
#include "ParallelProcessing/warp_reductions.h"
#include "Utilities/BuiltInTypes/vector_at.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h> // FLT_MAX

namespace Drafts
{
namespace LLM
{
namespace AttentionForward
{

//------------------------------------------------------------------------------
/// \param attention The output array for the attention scores. The effective
///   expected size is B * NH * T * T.
/// \param preattention The input array for the pre-attention scores. The
///   effective expected size is B * NH * T * T.
/// \param B Number of batches.
/// \param T Sequence length (number of tokens per sequence)
/// \param NH The number of attention heads.
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void attention_softmax_kernel1(
  FPType* attention,
  const FPType* preattention,
  const unsigned int B,
  const unsigned int T,
  const unsigned int NH)
{
  const unsigned int idx {blockIdx.x * blockDim.x + threadIdx.x};
  const unsigned int total_threads {B * T * NH};

  if (idx < total_threads)
  {
    const unsigned int h {idx % NH};
    const unsigned int t {(idx / NH) % T};
    // b \in 0, 1, ..., B - 1
    const unsigned int b {idx / (NH * T)};

    const FPType* preattention_bth {
      preattention + b * NH * T * T + h * T * T + t * T};
    FPType* attention_bth {attention + b * NH * T * T + h * T * T + t * T};

    // find maxval
    FPType maxval {-Numerics::Constants::get_infinity<FPType>()};
    for (unsigned int t2 {0}; t2 <= t; ++t2)
    {
      if (preattention_bth[t2] > maxval)
      {
        maxval = preattention_bth[t2];
      }
    }

    // calculate the exp and keep track of sum
    FPType expsum {0.0};
    for (unsigned int t2 {0}; t2 <= t; ++t2)
    {
      const FPType expv {
        Numerics::MathFunctions::get_exponential<FPType>(
          preattention_bth[t2] - maxval)};
      expsum += expv;
      attention_bth[t2] = expv;
    }
    const FPType expsum_inv {
      expsum == 0.0 ? static_cast<FPType>(0.0) :
        static_cast<FPType>(1.0) / expsum};

    // normalize to get the softmax
    for (unsigned int t2 {0}; t2 < T; ++t2)
    {
      if (t2 <= t)
      {
        attention_bth[t2] *= expsum_inv;
      }
      else
      {
        // causal attention mask. not strictly necessary to set to zero here
        // only doing this explicitly for debugging and checking to PyTorch
        attention_bth[t2] = 0.0;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// To launch a kernel for softmax_forward_kernel4,
/// __shared__ memory size is expected to be
/// 2 * warps_per_block * sizeof(FPType). If thread block size is 32 and number
/// of threads in a warp is 32, then warps_per_block is 1, typically.
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void softmax_forward_kernel4(
  FPType* out,
  const FPType* input,
  const unsigned int N,
  const unsigned int C)
{
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // same as kernel3, but can handle any block size (multiple of 32)
  // each row of C elements is handled by block_size threads
  // furthermore, each block_size threads get executed in warps of 32 threads

  // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
  // shared memory is used for inter-warp reduction
  extern __shared__ FPType shared[];
  const unsigned int idx {blockIdx.x};
  const unsigned int tid {threadIdx.x};
  const unsigned int warp_id {threadIdx.x / 32}; // warp index within a block
  const unsigned int lane_id {threadIdx.x % 32}; // thread index within a warp

  // the number of warps per block. recall that blockDim.x is block_size
  const unsigned int warps_per_block {blockDim.x / 32};

  // shared[] must be allocated to have 2 * warpsPerBlock elements
  // first half for max values, the second half for sum values
  FPType* max_values {shared};
  FPType* sum_values {&shared[warps_per_block]};

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  const FPType* x {input + idx * C};

  // first, thread coarsening by directly accessing global memory in series
  FPType max_value {-Numerics::Constants::get_infinity<FPType>()};
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    max_value = Numerics::MathFunctions::get_max<FPType>(max_value, x[i]);
  }

  // now within-warp reductions for maxval
  max_value = ParallelProcessing::warp_reduce_max(max_value);

  // the 0th thread of each warp writes the maxval of that warp to shared memory
  if (lane_id == 0)
  {
    max_values[warp_id] = max_value;
  }
  __syncthreads();

  // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
  if (tid == 0)
  {
    FPType val {max_values[tid]};
    for (unsigned int i {1}; i < warps_per_block; ++i)
    {
      val = Numerics::MathFunctions::get_max<FPType>(val, max_values[i]);
    }
    // store the final max in the first position
    max_values[0] = val;
  }
  __syncthreads();
  // broadcast the max to all threads
  FPType offset {max_values[0]};

  // compute expf and write the result to global memory
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    // subtract max for numerical stability
    out[idx * C + i] = Numerics::MathFunctions::get_exponential<FPType>(
      x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  FPType sum_value {0.0f};
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    sum_value += x[i];
  }
  // within-warp reduction for sumval
  sum_value = ParallelProcessing::warp_reduce_sum(sum_value);

  // write sumval to shared memory
  if (lane_id == 0)
  {
    sum_values[warp_id] = sum_value;
  }
  __syncthreads();

  // inter-thread reduction of sum
  if (tid == 0)
  {
    FPType value {sum_values[tid]};
    for (unsigned int i {1}; i < warps_per_block; ++i)
    {
      value += sum_values[i];
    }
    sum_values[0] = value;
  }
  __syncthreads();
  // broadcast the sum to all threads
  FPType sum {sum_values[0]};

  // divide the whole row by the sum
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    out[idx * C + i] = x[i] / sum;
  }
}

//------------------------------------------------------------------------------
/// This is exactly the same as softmax_forward_kernel4, but it reuses the
/// shared memory for both max and sum values..
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void softmax_forward_kernel4_reuse_shared_memory(
  FPType* out,
  const FPType* input,
  const unsigned int N,
  const unsigned int C)
{
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // same as kernel3, but can handle any block size (multiple of 32)
  // each row of C elements is handled by block_size threads
  // furthermore, each block_size threads get executed in warps of 32 threads

  // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
  // shared memory is used for inter-warp reduction
  extern __shared__ FPType shared[];
  const unsigned int idx {blockIdx.x};
  const unsigned int tid {threadIdx.x};
  const unsigned int warp_id {threadIdx.x / 32}; // warp index within a block
  const unsigned int lane_id {threadIdx.x % 32}; // thread index within a warp

  // the number of warps per block. recall that blockDim.x is block_size
  const unsigned int warps_per_block {blockDim.x / 32};

  // shared[] must be allocated to have 2 * warpsPerBlock elements
  // first half for max values, the second half for sum values
  FPType* max_or_sum_values {shared};

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  const FPType* x {input + idx * C};

  // first, thread coarsening by directly accessing global memory in series
  FPType max_value {-Numerics::Constants::get_infinity<FPType>()};
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    max_value = Numerics::MathFunctions::get_max<FPType>(max_value, x[i]);
  }

  // now within-warp reductions for maxval
  max_value = ParallelProcessing::warp_reduce_max(max_value);

  // the 0th thread of each warp writes the maxval of that warp to shared memory
  if (lane_id == 0)
  {
    max_or_sum_values[warp_id] = max_value;
  }
  __syncthreads();

  // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
  if (tid == 0)
  {
    FPType value {max_or_sum_values[tid]};
    for (unsigned int i {1}; i < warps_per_block; ++i)
    {
      value = Numerics::MathFunctions::get_max<FPType>(
        value,
        max_or_sum_values[i]);
    }
    // store the final max in the first position
    max_or_sum_values[0] = value;
  }
  __syncthreads();
  // broadcast the max to all threads
  FPType offset {max_or_sum_values[0]};

  // compute expf and write the result to global memory
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    // subtract max for numerical stability
    out[idx * C + i] = Numerics::MathFunctions::get_exponential<FPType>(
      x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  FPType sum_value {0.0f};
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    sum_value += x[i];
  }
  // within-warp reduction for sumval
  sum_value = ParallelProcessing::warp_reduce_sum(sum_value);

  // write sumval to shared memory
  if (lane_id == 0)
  {
    max_or_sum_values[warp_id] = sum_value;
  }
  __syncthreads();

  // inter-thread reduction of sum
  if (tid == 0)
  {
    FPType value {max_or_sum_values[tid]};
    for (unsigned int i {1}; i < warps_per_block; ++i)
    {
      value += max_or_sum_values[i];
    }
    max_or_sum_values[0] = value;
  }
  __syncthreads();
  // broadcast the sum to all threads
  FPType sum {max_or_sum_values[0]};

  // divide the whole row by the sum
  for (unsigned int i {tid}; i < C; i += blockDim.x)
  {
    out[idx * C + i] = x[i] / sum;
  }
}

//------------------------------------------------------------------------------
/// \param[in] T is the number of tokens in the sequence.
//------------------------------------------------------------------------------
template <typename FPType4, typename FPType>
__global__ void softmax_forward_kernel5(
  FPType* out,
  const FPType inverse_temperature,
  const FPType* input,
  const unsigned int N,
  const unsigned int T)
{
  // inp, out shape: (N, T, T), where N = B * NH
  // fuses the multiplication by scale inside attention
  // directly autoregressive, so we only compute the lower triangular part
  // uses the online softmax algorithm
  assert(T % 4  == 0);
  cooperative_groups::thread_block block {
    cooperative_groups::this_thread_block()};
  cooperative_groups::thread_block_tile<32> warp {
    cooperative_groups::tiled_partition<32>(block)};
  // micro-optimization: we iterate backwards so that
  // after the softmax backward operation completes, the cache retains the
  // part of the matrix close to the upper left corner, which benefits the
  // matmul operation that immediately follows.
  const unsigned int idx {
    (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() +
      warp.meta_group_rank()};

  if (idx >= N * T)
  {
    return;
  }
  const unsigned int sequence_position {idx % T};
  const unsigned int sequence_position_by_4 {sequence_position / 4};

  // one row of input, i.e. input[idx, :] of shape (T,)
  const FPType* x {input + idx * T};

  // not INF, so we don't get NaNs accidentally when subtracting two values.
  FPType max_value {-static_cast<FPType>(FLT_MAX)};
  FPType sum_value {0};

  const FPType4* x_vec {reinterpret_cast<const FPType4*>(x)};
  for (
    unsigned int i {warp.thread_rank()};
    i < sequence_position_by_4;
    i += warp.size())
  {
    const FPType4 v {x_vec[i]};
    FPType old_max_value {max_value};
    for(unsigned int k {0}; k < 4; ++k)
    {
      max_value = Numerics::MathFunctions::get_max<FPType>(
        max_value,
        Utilities::BuiltInTypes::vector_at<FPType4, FPType>(v, k));
    }
    sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (old_max_value - max_value));
    for (unsigned int k {0}; k < 4; ++k)
    {
      sum_value += Numerics::MathFunctions::get_exponential<FPType>(
        inverse_temperature * (
          Utilities::BuiltInTypes::vector_at<FPType4, FPType>(v, k) -
            max_value));
    }
  }

  if (4*sequence_position_by_4 + warp.thread_rank() <= sequence_position)
  {
    FPType old_max_value {max_value};
    max_value = Numerics::MathFunctions::get_max<FPType>(
      max_value,
      x[4*sequence_position_by_4 + warp.thread_rank()]);
    sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (old_max_value - max_value));
    sum_value += Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (
        x[4*sequence_position_by_4 + warp.thread_rank()] -
          max_value));
  }

  const FPType global_max_value {
    cooperative_groups::reduce(
      warp,
      max_value,
      cooperative_groups::greater<FPType>{})};
  sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
    inverse_temperature * (max_value - global_max_value));

  const FPType sum {
    cooperative_groups::reduce(
      warp,
      sum_value,
      cooperative_groups::plus<FPType>{})};
  const FPType norm {1.0f / sum};

  // divide the whole row by the sum
  for (
    unsigned int i {warp.thread_rank()};
    i <= sequence_position;
    i += warp.size())
  {
    // recalculation is faster than doing the round-trip through memory.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldcs#load-functions-using-cache-hints
    // T __ldcs(const T* address);
    // returns data of type T located at address address.
    const FPType ev {
      Numerics::MathFunctions::get_exponential<FPType>(
        inverse_temperature * (__ldcs(x + i) - global_max_value))};
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldcs#store-functions-using-cache-hints
    // void __stcs(T* address, T value);
    // stores value argument of type T to location at address address.
    __stcs(out + idx * T + i, ev * norm);
  }
}

} // namespace AttentionForward
} // namespace LLM
} // namespace Drafts

#endif // DRAFTS_LLM_ATTENTION_FORWARD_SOFTMAX_H
