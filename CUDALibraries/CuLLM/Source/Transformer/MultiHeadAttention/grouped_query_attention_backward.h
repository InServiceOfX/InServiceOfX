#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_BACKWARD_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_BACKWARD_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "StreamManagement/Stream.h"
#include "Transformer/Attention/flash_attention_backward.h"
#include "Transformer/MultiHeadAttention/multi_head_attention_backward.h"
#include "Transformer/MultiHeadAttention/split_qkv_heads.h"

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Grouped-query attention backward pass: gradients of a scalar loss
/// through Y = GQA(X) w.r.t. the input X and both learned weight matrices
/// W_qkv = [W^Q | W^K | W^V] (∈ R^{d_model × (NH + 2·NKV)·kHeadDim}) and
/// W^O. The chain mirrors multi_head_attention_backward with two
/// group-specific insertions (see the remark on Multi-query and
/// grouped-query attention in FlashAttention.tex: "gradients dK, dV for
/// the g heads sharing a (K, V) pair must be summed"):
///
///   1. merge_heads          — recompute H from the saved per-head output.
///   2. linear_map_backward  — dW^O = H^⊤ dY and dH = dY W^{O⊤}
///                             (query side is unchanged by grouping).
///   3. split_heads          — dH → per-head dO.
///   4. flash_attention_backward with (num_heads, kv_group_size) — dQ per
///                             query head; dK/dV as per-query-head
///                             PARTIALS in (B·NH, T, kHeadDim) buffers
///                             (single-writer, no atomics).
///   5. reduce_grouped_kv_gradients — the group sum: partials
///                             (B·NH, ...) → true dK/dV (B·NKV, ...).
///   6. merge_grouped_qkv_heads — dQ + reduced dK/dV → d(qkv) in the fused
///                             (B·T, (NH + 2·NKV)·kHeadDim) layout.
///   7. linear_map_backward  — dW_qkv = X^⊤ d(qkv), dX = d(qkv) W_qkv^⊤,
///                             with the fused width as the output-column
///                             count (the GEMM helpers are shape-generic).
///
/// kv_group_size = 1 computes exactly multi_head_attention_backward (the
/// reduction is then a copy and the fused layout coincides with [Q|K|V]).
///
/// Required forward activations (saved by grouped_query_attention with a
/// non-null logsumexp): input X, queries (B·NH, T, kHeadDim), keys/values
/// (B·NKV, T, kHeadDim), attention_output (B·NH, T, kHeadDim), logsumexp
/// (B·NH, T).
///
/// Caller-allocated outputs:
///   gradient_input          — (B·T, d_model), dX
///   gradient_qkv_weights    — (d_model, (NH + 2·NKV)·kHeadDim), dW_qkv
///   gradient_output_weights — (d_model, d_model), dW^O
///
/// Caller-allocated workspaces:
///   concat_workspace            — (B·T, d_model), recomputed H
///   gradient_concat_workspace   — (B·T, d_model), dH
///   gradient_attention_output   — (B·NH, T, kHeadDim), per-head dO
///   gradient_queries            — (B·NH, T, kHeadDim), dQ
///   partial_gradient_keys/values— (B·NH, T, kHeadDim) each, pre-reduction
///   gradient_keys/values        — (B·NKV, T, kHeadDim) each, group-summed
///   gradient_qkv_workspace      — (B·T, (NH + 2·NKV)·kHeadDim), d(qkv)
///   row_dots_workspace          — (B·NH, T)
///
/// kCausal must match the forward call.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
bool grouped_query_attention_backward(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* gradient_input,
  T* gradient_qkv_weights,
  T* gradient_output_weights,
  T* concat_workspace,
  T* gradient_concat_workspace,
  T* gradient_attention_output,
  T* gradient_queries,
  T* partial_gradient_keys,
  T* partial_gradient_values,
  T* gradient_keys,
  T* gradient_values,
  T* gradient_qkv_workspace,
  T* row_dots_workspace,
  const T* gradient_output,
  const T* input,
  const T* qkv_weight_matrix,
  const T* output_weight_matrix,
  const T* queries,
  const T* keys,
  const T* values,
  const T* attention_output,
  const T* logsumexp,
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int sequence_length)
{
  static_assert(
    kHeadDim % 32 == 0,
    "grouped_query_attention_backward requires kHeadDim to be a multiple of "
    "32 (warp size), matching the forward pass constraint.");

  if (num_heads % kv_group_size != 0)
  {
    return false;
  }
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHeadDim};
  const int fused_width {(num_heads + 2 * num_kv_heads) * kHeadDim};
  const int num_tokens {batch_size * sequence_length};

  constexpr int kThreadsPerBlock {256};
  const long long total_query_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};
  const int number_of_blocks {static_cast<int>(
    (total_query_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  // 1. Recompute H (query side — identical to the MHA backward).
  merge_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    concat_workspace,
    attention_output,
    batch_size,
    num_heads,
    sequence_length);

  // 2. dW^O = H^⊤ dY, dH = dY W^{O⊤}.
  if (!linear_map_backward<T>(
    handle,
    stream,
    gradient_concat_workspace,
    gradient_output_weights,
    gradient_output,
    concat_workspace,
    output_weight_matrix,
    num_tokens,
    d_model,
    d_model))
  {
    return false;
  }

  // 3. dH → per-head dO.
  split_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    gradient_attention_output,
    gradient_concat_workspace,
    batch_size,
    num_heads,
    sequence_length);

  // 4. dQ per query head; dK/dV as per-query-head partials.
  Attention::flash_attention_backward<T, kHeadDim, kWarpsPerBlock, kCausal>(
    gradient_queries,
    partial_gradient_keys,
    partial_gradient_values,
    row_dots_workspace,
    queries,
    keys,
    values,
    attention_output,
    gradient_attention_output,
    logsumexp,
    sequence_length,
    batch_size * num_heads,
    num_heads,
    kv_group_size);

  // 5. The group sum: dK_{kv} = Σ_j dK_{query head kv·g + j} (same for dV).
  const long long total_kv_elements {
    static_cast<long long>(batch_size) * num_kv_heads * sequence_length *
      kHeadDim};
  const int kv_blocks {static_cast<int>(
    (total_kv_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};
  Attention::reduce_grouped_kv_gradients<T, kHeadDim>
    <<<kv_blocks, kThreadsPerBlock>>>(
      gradient_keys,
      gradient_values,
      partial_gradient_keys,
      partial_gradient_values,
      batch_size,
      num_heads,
      kv_group_size,
      sequence_length);

  // 6. dQ + group-summed dK/dV → d(qkv) in the fused grouped layout.
  merge_grouped_qkv_heads<T, kHeadDim>
    <<<number_of_blocks, kThreadsPerBlock>>>(
      gradient_qkv_workspace,
      gradient_queries,
      gradient_keys,
      gradient_values,
      batch_size,
      num_heads,
      kv_group_size,
      sequence_length);

  // 7. dW_qkv = X^⊤ d(qkv), dX = d(qkv) W_qkv^⊤ over the fused width.
  return linear_map_backward<T>(
    handle,
    stream,
    gradient_input,
    gradient_qkv_weights,
    gradient_qkv_workspace,
    input,
    qkv_weight_matrix,
    num_tokens,
    d_model,
    fused_width);
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_BACKWARD_H
