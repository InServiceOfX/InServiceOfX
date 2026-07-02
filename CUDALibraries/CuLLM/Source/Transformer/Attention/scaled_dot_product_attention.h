#ifndef TRANSFORMER_ATTENTION_SCALED_DOT_PRODUCT_ATTENTION_H
#define TRANSFORMER_ATTENTION_SCALED_DOT_PRODUCT_ATTENTION_H

#include "Transformer/Attention/attention_scores.h"
#include "Transformer/Attention/attention_weighted_values.h"
#include "Transformer/Softmax/softmax_warp_fold_reduce.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// Scaled dot-product attention as the composition of three maps
/// (see the section on Scaled Dot-Product Attention in FlashAttention.tex):
///
///   Att(Q, K, V) := softmax(Q K^⊤ / √d_k) V  ∈ R^{n×d_v},
///
/// launched as one kernel per map:
///
///   1. S := Q K^⊤ / √d_k          (attention_scores)
///   2. P := softmax_rows(S)       (softmax_warp_fold_reduce)
///   3. O := P V                   (attention_weighted_values)
///
/// This is *standard* attention: both S and P — n×n matrices — are
/// materialized in HBM and re-read by the next stage, so HBM traffic is
/// Θ(n² + n·d) (see the section on IO Complexity of Standard Attention in
/// FlashAttention.tex). Since attention is memory-bound at these sizes, that
/// n² term dominates the runtime. This implementation is the correctness and
/// IO baseline; FlashAttention removes the n² traffic by folding K/V tiles
/// through the AttentionAccumulator merge monoid inside a single kernel,
/// never writing S or P to HBM.
///
/// scores_workspace and weights_workspace are caller-allocated device buffers
/// of sequence_length² elements each (S and P respectively); they are kept
/// separate and exposed so tests can inspect the intermediate matrices —
/// exactly the intermediates FlashAttention eliminates.
///
/// kHeadDim = d_k = d_v must be a compile-time constant (matching
/// attention_scores, attention_weighted_values, and
/// AttentionAccumulator<AccT, kHeadDim>).
///
/// block_size must be a multiple of the warp size (32) for the softmax
/// stage's warp partitioning.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
void scaled_dot_product_attention(
  T* output,
  T* scores_workspace,
  T* weights_workspace,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length,
  const int block_size = 128)
{
  // 1. S = Q K^⊤ / √d_k — one block per query row.
  attention_scores<T, kHeadDim><<<sequence_length, block_size>>>(
    scores_workspace,
    queries,
    keys,
    sequence_length);

  // 2. P = softmax_rows(S) — one warp per row of S, blockDim / 32 warps per
  // block, so ceil(n / warps_per_block) blocks cover all n rows.
  constexpr int WARP_SIZE {32};
  const int warps_per_block {block_size / WARP_SIZE};
  const int softmax_blocks {
    (sequence_length + warps_per_block - 1) / warps_per_block};
  Softmax::softmax_warp_fold_reduce<T><<<softmax_blocks, block_size>>>(
    weights_workspace,
    scores_workspace,
    sequence_length,
    sequence_length);

  // 3. O = P V — one block per output row.
  attention_weighted_values<T, kHeadDim><<<sequence_length, block_size>>>(
    output,
    weights_workspace,
    values,
    sequence_length);
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_SCALED_DOT_PRODUCT_ATTENTION_H
