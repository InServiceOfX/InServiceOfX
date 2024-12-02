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

} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_H