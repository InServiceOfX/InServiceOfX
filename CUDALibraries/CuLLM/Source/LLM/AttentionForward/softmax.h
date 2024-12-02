#ifndef LLM_ATTENTION_FORWARD_SOFTMAX_H
#define LLM_ATTENTION_FORWARD_SOFTMAX_H

#include "Numerics/MathFunctions.h"
#include "ParallelProcessing/warp_reductions.h"

namespace LLM
{
namespace AttentionForward
{

//------------------------------------------------------------------------------
/// Based upon softmax_forward_kernel5(..) in
/// https://github.com/karpathy/llm.c/blob/master/llmc/attention.cuh#L85
/// \param[in] T is the number of tokens in the sequence.
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void softmax_forward(
  FPType* out,
  const FPType inverse_temperature,
  const FPType* input,
  const unsigned int N,
  const unsigned int T)
{
  // TODO: Is this or global __constant__ better?
  static constexpr unsigned int warp_size {32};

  // inp, out shape: (N, T, T), where N = B * NH
  // fuses the multiplication by scale inside attention
  // directly autoregressive, so we only compute the lower triangular part
  // uses the online softmax algorithm
  //assert(T % 4  == 0);
  const unsigned int lane_id {threadIdx.x % warp_size};
  const unsigned int warp_id {threadIdx.x / warp_size};
  const unsigned int number_of_warps {blockDim.x / warp_size};

  // micro-optimization: we iterate backwards so that
  // after the softmax backward operation completes, the cache retains the
  // part of the matrix close to the upper left corner, which benefits the
  // matmul operation that immediately follows.
  const unsigned int idx {
    (gridDim.x - blockIdx.x - 1) * number_of_warps + warp_id};

  if (idx >= N * T)
  {
    return;
  }
  const unsigned int sequence_position {idx % T};
  const unsigned int sequence_position_by_4 {sequence_position / 4};

  // one row of input, i.e. input[idx, :] of shape (T,)
  const FPType* x {input + idx * T};

  // not INF, so we don't get NaNs accidentally when subtracting two values.
  // In Karpathy's implementation, they avoid #include <float.h> and use this
  // value.
  FPType max_value {
    -static_cast<FPType>(340282346638528859811704183484516925440.0f)};
  FPType sum_value {0};

  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__builtin_assume_aligned#builtin-assume-aligned
  // void * __builtin_assume_aligned(const void *exp, size_t align)
  // Allows compiler to assume argument pointer is aligned to at least align
  // bytes, and returns argument pointer.
  constexpr unsigned int alignment {sizeof(FPType) * 4};

  const FPType* x_aligned {
    reinterpret_cast<const FPType*>(__builtin_assume_aligned(x, alignment))};
  for (
    unsigned int i {lane_id};
    i < sequence_position_by_4;
    i += warp_size)
  {
    FPType regarray[4];
    for (unsigned int k {0}; k < 4; ++k)
    {
      regarray[k] = static_cast<FPType>(x_aligned[4*i + k]);
    }
    const FPType old_max_value {max_value};
    for (unsigned int k {0}; k < 4; ++k)
    {
      max_value = Numerics::MathFunctions::get_max<FPType>(
        max_value,
        regarray[k]);
    }
    sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (old_max_value - max_value));
    for (unsigned int k {0}; k < 4; ++k)
    {
      sum_value += Numerics::MathFunctions::get_exponential<FPType>(
        inverse_temperature * (regarray[k] - max_value));
    }
  }

  if (4*sequence_position_by_4 + lane_id <= sequence_position)
  {
    const FPType old_max_value {max_value};
    max_value = Numerics::MathFunctions::get_max<FPType>(
      max_value,
      static_cast<FPType>(x[4*sequence_position_by_4 + lane_id]));
    sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (old_max_value - max_value));
    sum_value += Numerics::MathFunctions::get_exponential<FPType>(
      inverse_temperature * (
        static_cast<FPType>(x[4*sequence_position_by_4 + lane_id]) -
          max_value));
  }

  const FPType global_max_value {
    ParallelProcessing::warp_reduce_max<FPType>(max_value)};
  sum_value *= Numerics::MathFunctions::get_exponential<FPType>(
    inverse_temperature * (max_value - global_max_value));

  const FPType sum {ParallelProcessing::warp_reduce_sum<FPType>(sum_value)};
  const FPType norm {1.0f / sum};

  // divide the whole row by the sum
  for (unsigned int i {lane_id}; i <= sequence_position; i += warp_size)
  {
    // recalculation is faster than doing the round-trip through memory.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldcs#load-functions-using-cache-hints
    // T __ldcs(const T* address);
    // returns data of type T located at address address.
    const FPType ev {
      Numerics::MathFunctions::get_exponential<FPType>(
        inverse_temperature * (
          static_cast<FPType>(__ldcs(x + i)) - global_max_value))};
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__ldcs#store-functions-using-cache-hints
    // void __stcs(T* address, T value);
    // stores value argument of type T to location at address address.
    __stcs(out + idx * T + i, static_cast<FPType>(ev * norm));
  }
}

} // namespace AttentionForward
} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_SOFTMAX_H