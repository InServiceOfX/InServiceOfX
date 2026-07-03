#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_DISPATCH_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_DISPATCH_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#if defined(CULLM_HAS_CUTLASS)
#include "Transformer/Attention/flash_attention_cute.h"
#endif
#include "Transformer/Attention/flash_attention_warp_cooperative.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// Runtime/compile-time dispatch for the best available FlashAttention forward
/// kernel with the same public contract as flash_attention_warp_cooperative.
///
/// The CuTe/CUTLASS implementation is currently a narrow but faster engine:
/// __half I/O, kHeadDim = 64, kWarpsPerBlock = 4, ordinary per-head K/V
/// layout, and sm_80+ at runtime. Every other shape and build configuration
/// falls back to the existing warp-cooperative kernel. This keeps the full MHA
/// API stable while exposing the CUTLASS path for the production-eligible
/// case.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock>
bool flash_attention_forward_dispatch_uses_cute()
{
#if defined(CULLM_HAS_CUTLASS)
  if constexpr (
    std::is_same_v<T, __half> && kHeadDim == 64 && kWarpsPerBlock == 4)
  {
    int device {};
    if (cudaGetDevice(&device) != cudaSuccess)
    {
      return false;
    }

    cudaDeviceProp properties {};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
      return false;
    }

    return properties.major >= 8;
  }
#endif
  return false;
}

template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
void flash_attention_forward_dispatch(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length,
  const int number_of_batch_heads = 1)
{
#if defined(CULLM_HAS_CUTLASS)
  if constexpr (
    std::is_same_v<T, __half> && kHeadDim == 64 && kWarpsPerBlock == 4)
  {
    if (flash_attention_forward_dispatch_uses_cute<
      T,
      kHeadDim,
      kWarpsPerBlock>())
    {
      flash_attention_cute<T, kHeadDim, kWarpsPerBlock, kCausal>(
        output,
        logsumexp,
        queries,
        keys,
        values,
        sequence_length,
        number_of_batch_heads);
      return;
    }
  }
#endif

  flash_attention_warp_cooperative<T, kHeadDim, kWarpsPerBlock, kCausal>(
    output,
    logsumexp,
    queries,
    keys,
    values,
    sequence_length,
    number_of_batch_heads);
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_DISPATCH_H
