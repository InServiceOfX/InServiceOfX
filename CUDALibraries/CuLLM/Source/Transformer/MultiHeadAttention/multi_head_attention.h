#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "StreamManagement/Stream.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"
#include "Transformer/MultiHeadAttention/output_linear_map.h"
#include "Transformer/MultiHeadAttention/qkv_linear_maps.h"

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Multi-head attention forward pass (see the section on Multi-Head
/// Attention in FlashAttention.tex):
///
///   MHA(y) := [head_1 | ... | head_h] W^O,
///   head_ℓ := Att(yW^Q_ℓ, yW^K_ℓ, yW^V_ℓ),
///
/// realised as three stages, each already independently defined and tested
/// elsewhere in this library:
///
///   1. qkv_linear_maps — one fused cuBLASLt GEMM applies the learned
///      right-multiplication linear maps for every head at once
///      (y W_qkv), then split_qkv_heads gathers the result into
///      per-head-contiguous tensors.
///   2. flash_attention_warp_cooperative — the attention core, run once per
///      (batch, head) slice via its existing blockIdx.y batching (see the
///      section on The FlashAttention Algorithm); this is exactly the
///      independence multihead_flash_attention_tests.cu verifies.
///   3. output_linear_map — merge_heads concatenates the per-head outputs
///      back to (B·T, d_model), then one cuBLASLt GEMM applies the learned
///      right-multiplication linear map defined by W^O.
///
/// All workspace buffers are caller-allocated (this library's convention
/// throughout — see e.g. scaled_dot_product_attention.h's scores/weights
/// workspaces), sized as follows for batch_size B, num_heads NH, head
/// dimension kHeadDim = d_k = d_v, d_model = NH·kHeadDim, sequence_length T:
///
///   qkv_workspace       — (B·T, 3·d_model)
///   queries/keys/values — (B·NH, T, kHeadDim) each
///   attention_output    — (B·NH, T, kHeadDim)
///   concat_workspace    — (B·T, d_model)
///   output              — (B·T, d_model), the final MHA(y)
///
/// input is (B·T, d_model), y of the definition above, tokens of all batch
/// elements stacked as rows. qkv_weight_matrix is (d_model, 3·d_model) and
/// output_weight_matrix is (d_model, d_model); see qkv_linear_maps.h and
/// split_qkv_heads.h for the exact per-head column layout within each.
///
/// kWarpsPerBlock is flash_attention_warp_cooperative's tile parameter (one
/// warp per query row, kWarpsPerBlock rows per thread block); kCausal
/// selects masked (decoder self-attention) vs. unmasked attention.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
bool multi_head_attention(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* output,
  T* qkv_workspace,
  T* queries,
  T* keys,
  T* values,
  T* attention_output,
  T* concat_workspace,
  const T* input,
  const T* qkv_weight_matrix,
  const T* output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  // Surface the constraint here, at the API boundary, rather than as a deep
  // template error inside the attention core. The warp-cooperative kernel
  // shards each õ row across the 32 lanes of a warp (kFragment components
  // per lane), so kHeadDim must divide evenly. Real transformer head dims
  // (32/64/128) all satisfy this; for other sizes use
  // scaled_dot_product_attention (the non-flash baseline) or pad the head
  // dimension.
  static_assert(
    kHeadDim % 32 == 0,
    "multi_head_attention requires kHeadDim to be a multiple of 32 (warp "
    "size): flash_attention_warp_cooperative shards each output row across "
    "the lanes of one warp. Use kHeadDim of 32/64/128, pad the head "
    "dimension, or call scaled_dot_product_attention instead.");

  if (!qkv_linear_maps<T, kHeadDim>(
    handle,
    stream,
    queries,
    keys,
    values,
    qkv_workspace,
    input,
    qkv_weight_matrix,
    batch_size,
    num_heads,
    sequence_length))
  {
    return false;
  }

  // head_ℓ = Att(Q_ℓ, K_ℓ, V_ℓ) for every (batch, head) slice at once.
  Attention::flash_attention_warp_cooperative<T, kHeadDim, kWarpsPerBlock, kCausal>(
    attention_output,
    nullptr,
    queries,
    keys,
    values,
    sequence_length,
    batch_size * num_heads);

  return output_linear_map<T, kHeadDim>(
    handle,
    stream,
    output,
    concat_workspace,
    attention_output,
    output_weight_matrix,
    batch_size,
    num_heads,
    sequence_length);
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_H
