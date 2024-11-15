#ifndef LLM_ATTENTION_FORWARD_H
#define LLM_ATTENTION_FORWARD_H

#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits

namespace LLM
{

//------------------------------------------------------------------------------
/// \brief Compute the pre-attention scores.
/// \param preattention The output array for the pre-attention scores.
/// \param inp The input array containing the query, key, and value tensors.
/// \param B Number of batches..
/// \param T Sequence length (number of tokens per sequence)
/// \param C Total feature dimension (combined across all attention heads)
/// \param NH The number of attention heads.
//------------------------------------------------------------------------------
template <typename FPType>
__global__ void attention_query_key_kernel1(
  FPType* preattention,
  const FPType* inp,
  const int B,
  const int T,
  const int C,
  const int NH)
{
  const std::size_t idx {blockIdx.x * blockDim.x + threadIdx.x};
  const std::size_t total_number_of_threads {B * NH * T * T};

  if (idx < total_number_of_threads)
  {
    const std::size_t t2 {idx % T};
    const std::size_t t {(idx / T) % T};
    if (t2 > t)
    {
      // autoregressive mask
      preattention[idx] = -std::numeric_limits<FPType>::infinity();
      return;
    }

    // attention head index
    const std::size_t h {(idx / (T * T)) % NH};
    // batch index
    const std::size_t b {idx / (NH * T * T)};

    const int C3 {C * 3};
    // head size, dimensionality of single attention head, d.
    const int hs {C / NH};

    const T* query_t {inp + b * T * C3 + t * C3 + h * hs};
    // +C because it's a key.
    const T* key_t2 {inp + b * T * C3 + t2 * C3 + h * hs + C};

    // (query_t) dot (key_t2)
    T value {0.0f};
    for (int i {0}; i < hs; ++i)
    {
      value += query_t[i] * key_t2[i];
    }

    value *= 1.0 / sqrt(hs);

    preattention[idx] = value;
  }
}

} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_H