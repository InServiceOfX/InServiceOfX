#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "StreamManagement/Stream.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"
#include "Transformer/MultiHeadAttention/output_linear_map.h"
#include "Transformer/MultiHeadAttention/split_qkv_heads.h"

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Grouped-query attention forward pass (see the remark on Multi-query and
/// grouped-query attention in FlashAttention.tex): multi-head attention
/// with the key/value weight matrices shared across groups of
/// kv_group_size query heads,
///
///   head_ℓ := Att(y W^Q_ℓ, y W^K_{⌈ℓ/g⌉}, y W^V_{⌈ℓ/g⌉}),   g := kv_group_size,
///
/// so only NKV = num_heads/g distinct (K, V) pairs are computed and stored.
/// g = 1 recovers multi_head_attention exactly; g = num_heads is multi-query
/// attention (one shared pair). The K/V projection — and the K/V cache a
/// decoder keeps at autoregressive inference time — shrinks by the factor g
/// without changing the per-head attention computation.
///
/// Same three-stage structure as multi_head_attention:
///   1. one fused cuBLASLt GEMM y W_qkv with
///      W_qkv = [W^Q | W^K | W^V] ∈ R^{d_model × (NH + 2·NKV)·kHeadDim},
///      then split_grouped_qkv_heads gathers per-head/per-KV-head tensors;
///   2. flash_attention_warp_cooperative with the group-aware K/V slice
///      map (query slice b·NH + h reads K/V slice b·NKV + h/g);
///   3. merge_heads + output linear map, identical to MHA (the query-side
///      head count is unchanged).
///
/// Buffer shapes (B = batch_size, NH = num_heads, NKV = NH/g, T =
/// sequence_length, d_model = NH·kHeadDim):
///   qkv_workspace       — (B·T, (NH + 2·NKV)·kHeadDim)
///   queries             — (B·NH, T, kHeadDim)
///   keys/values         — (B·NKV, T, kHeadDim) each   ← g× smaller than MHA
///   attention_output    — (B·NH, T, kHeadDim)
///   concat_workspace    — (B·T, d_model)
///   output              — (B·T, d_model)
///   qkv_weight_matrix   — (d_model, (NH + 2·NKV)·kHeadDim)
///   output_weight_matrix— (d_model, d_model)
///
/// The backward pass is NOT implemented for g > 1: the gradients dK, dV of
/// the g heads sharing a (K, V) pair must be summed (see the tex remark),
/// which multi_head_attention_backward's per-slice kernels do not do. Use
/// g = 1 (i.e. multi_head_attention) for training paths until that
/// reduction exists.
///
/// num_heads must be divisible by kv_group_size.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
bool grouped_query_attention(
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
  const int kv_group_size,
  const int sequence_length,
  T* logsumexp = nullptr)
{
  static_assert(
    kHeadDim % 32 == 0,
    "grouped_query_attention requires kHeadDim to be a multiple of 32 (warp "
    "size): flash_attention_warp_cooperative shards each output row across "
    "the lanes of one warp.");

  if (num_heads % kv_group_size != 0)
  {
    return false;
  }
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHeadDim};
  const int fused_width {(num_heads + 2 * num_kv_heads) * kHeadDim};
  const int num_tokens {batch_size * sequence_length};

  // Fused projection y W_qkv, same row-major-via-column-major relabelling
  // as qkv_linear_maps.h with the narrower output width.
  {
    cuBLASWrappers::MatrixMultiplication::Setup<T> setup(
      fused_width, num_tokens, d_model);
    if (!setup.setup(handle))
    {
      return false;
    }
    cuBLASWrappers::MatrixMultiplication::LtMatrixMultiplication<T> matmul{};
    if (!matmul(
      handle,
      setup.descriptor_,
      setup.layouts_,
      setup.heuristic_,
      stream,
      setup.workspace_,
      qkv_weight_matrix,
      input,
      nullptr,
      qkv_workspace))
    {
      return false;
    }
  }

  constexpr int kThreadsPerBlock {256};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};
  const int number_of_blocks {static_cast<int>(
    (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  split_grouped_qkv_heads<T, kHeadDim>
    <<<number_of_blocks, kThreadsPerBlock>>>(
      queries,
      keys,
      values,
      qkv_workspace,
      batch_size,
      num_heads,
      kv_group_size,
      sequence_length);

  // head_ℓ = Att(Q_ℓ, K_{⌈ℓ/g⌉}, V_{⌈ℓ/g⌉}) for every (batch, query head)
  // slice; the kernel maps each query slice to its group's K/V slice.
  Attention::flash_attention_warp_cooperative<
    T, kHeadDim, kWarpsPerBlock, kCausal>(
      attention_output,
      logsumexp,
      queries,
      keys,
      values,
      sequence_length,
      batch_size * num_heads,
      num_heads,
      kv_group_size);

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

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_GROUPED_QUERY_ATTENTION_H
