#ifndef LLM_ATTENTION_FORWARD_H
#define LLM_ATTENTION_FORWARD_H

#include "Numerics/MathFunctions.h"
#include "Numerics/Constants/get_infinity.h"

namespace LLM
{

//------------------------------------------------------------------------------
/// \brief Compute the pre-attention scores.
/// \param preattention The output array for the pre-attention scores. The
///   effective expected size is B * NH * T * T.
/// \param input The input array containing the query, key, and value tensors. The
///   effective expected size is B * T * C * 3.
/// \param B Number of batches.
/// \param T Sequence length (number of tokens per sequence)
/// \param C Total feature dimension (combined across all attention heads)
/// \param NH The number of attention heads.
/// See
/// https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/dev/cuda/attention_forward.cu#L160
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void attention_query_key_kernel1(
  FPType* preattention,
  const FPType* input,
  const int B,
  const int T,
  const int C,
  const int NH)
{
  const size_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  const size_t total_number_of_threads {
    static_cast<size_t>(B) * NH * T * T};

  if (idx < total_number_of_threads)
  {
    const size_t t2 {idx % T};
    const size_t t {(idx / T) % T};
    if (t2 > t)
    {
      // autoregressive mask
      preattention[idx] = -Numerics::Constants::get_infinity<FPType>();
      return;
    }

    // attention head index
    const size_t h {(idx / (T * T)) % NH};
    // batch index
    const size_t b {idx / (NH * T * T)};

    const int C3 {C * 3};
    // head size, dimensionality of single attention head, d.
    const int hs {C / NH};

    const FPType* query_t {input + b * T * C3 + t * C3 + h * hs};
    // +C because it's a key.
    const FPType* key_t2 {input + b * T * C3 + t2 * C3 + h * hs + C};

    // (query_t) dot (key_t2)
    FPType value {0.0f};
    for (int i {0}; i < hs; ++i)
    {
      value += query_t[i] * key_t2[i];
    }

    value *= 1.0 / Numerics::MathFunctions::get_sqrt<FPType>(hs);

    preattention[idx] = value;
  }
}

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
  const int B,
  const int T,
  const int NH)
{
  const size_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  const size_t total_threads {static_cast<size_t>(B) * T * NH};

  if (idx < total_threads)
  {
    const size_t h {idx % NH};
    const size_t t {(idx / NH) % T};
    // b \in 0, 1, ..., B - 1
    const size_t b {idx / (NH * T)};

    const FPType* preattention_bth {
      preattention + b * NH * T * T + h * T * T + t * T};
    FPType* attention_bth {attention + b * NH * T * T + h * T * T + t * T};

    // find maxval
    FPType maxval {-Numerics::Constants::get_infinity<FPType>()};
    for (size_t t2 {0}; t2 <= t; ++t2)
    {
      if (preattention_bth[t2] > maxval)
      {
        maxval = preattention_bth[t2];
      }
    }

    // calculate the exp and keep track of sum
    FPType expsum {0.0};
    for (size_t t2 {0}; t2 <= t; ++t2)
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
    for (size_t t2 {0}; t2 < T; ++t2)
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

} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_H