#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_SPLIT_QKV_HEADS_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_SPLIT_QKV_HEADS_H

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Splits the output of the fused QKV linear-map GEMM into three
/// per-head-contiguous tensors, ready for the attention core (see the
/// section on Multi-Head Attention in FlashAttention.tex).
///
/// The fused GEMM applies all learned Q/K/V right-multiplication linear maps
/// at once: qkv := X W_qkv, where X ∈ R^{(B·T)×d_model} stacks all tokens of
/// all batch elements as rows, and
///
///   W_qkv := [ W^Q | W^K | W^V ] ∈ R^{d_model × 3·d_model},
///   W^Q   := [ W^Q_0 | W^Q_1 | ... | W^Q_{NH-1} ] ∈ R^{d_model × d_model},
///
/// (similarly W^K, W^V), each W^Q_ℓ ∈ R^{d_model × kHeadDim} the per-head
/// weight matrix of Definition~(Multi-Head Attention). So row-major qkv has,
/// for token row r, the layout
///
///   qkv[r, :] = [Q_0(r) ... Q_{NH-1}(r) | K_0(r) ... K_{NH-1}(r)
///                | V_0(r) ... V_{NH-1}(r)],
///
/// each Q_ℓ(r)/K_ℓ(r)/V_ℓ(r) a length-kHeadDim block — one wide GEMM
/// instead of 3·NH narrow ones (better arithmetic intensity, one cuBLASLt
/// call). This kernel is a pure gather: it does not touch the GEMM's
/// numerics, only its memory layout.
///
/// Output layout: queries, keys, values are each row-major
/// (B·NH, T, kHeadDim) — i.e. one contiguous (T, kHeadDim) slice per
/// (batch, head) pair, exactly the layout flash_attention's
/// blockIdx.y-indexed slices expect (see flash_attention_forward.h).
///
/// kHeadDim must be a compile-time constant, matching every other
/// Attention/ kernel. num_heads is a runtime parameter (it only affects the
/// gather's index arithmetic, not any per-thread register array size).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void split_qkv_heads(
  T* queries,
  T* keys,
  T* values,
  const T* qkv,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    // Row r = b·T + t of qkv; head h's block starts at column h·kHeadDim
    // within each of the three d_model-wide segments.
    const long long qkv_row {
      (static_cast<long long>(b) * sequence_length + t) * 3 * d_model};
    const long long head_offset {
      static_cast<long long>(h) * kHeadDim + d};

    const long long out_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};

    queries[out_index] = qkv[qkv_row + head_offset];
    keys[out_index] = qkv[qkv_row + d_model + head_offset];
    values[out_index] = qkv[qkv_row + 2 * d_model + head_offset];
  }
}

//------------------------------------------------------------------------------
/// Inverse of split_qkv_heads: concatenates per-head attention output back
/// into the row-major (B·T, d_model) layout the output linear-map GEMM
/// expects (see the concatenation in the section on Multi-Head Attention:
/// MHA(y) := [head_1 | ... | head_h] W^O).
///
/// Input per_head_output is row-major (B·NH, T, kHeadDim) — the layout
/// flash_attention writes. Output concatenated is row-major (B·T, d_model),
/// with head h's slice placed at columns [h·kHeadDim, (h+1)·kHeadDim).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void merge_heads(
  T* concatenated,
  const T* per_head_output,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    const long long in_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};
    const long long out_index {
      (static_cast<long long>(b) * sequence_length + t) * d_model +
        h * kHeadDim + d};

    concatenated[out_index] = per_head_output[in_index];
  }
}

//------------------------------------------------------------------------------
/// Grouped-query variant of split_qkv_heads (see the remark on Multi-query
/// and grouped-query attention in FlashAttention.tex): the fused GEMM output
/// has row layout
///
///   qkv[r, :] = [Q_0(r) ... Q_{NH-1}(r) | K_0(r) ... K_{NKV-1}(r)
///                | V_0(r) ... V_{NKV-1}(r)],
///
/// where NKV = num_heads / kv_group_size is the number of *distinct* K/V
/// heads — the fused weight matrix is (d_model, (NH + 2·NKV)·kHeadDim),
/// shrinking the K/V projection (and the inference-time K/V cache) by the
/// group factor. Queries gather to (B·NH, T, kHeadDim) exactly as
/// split_qkv_heads; keys and values gather to (B·NKV, T, kHeadDim).
/// kv_group_size = 1 reproduces split_qkv_heads' layout bit for bit;
/// kv_group_size = num_heads is multi-query attention (one shared pair).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void split_grouped_qkv_heads(
  T* queries,
  T* keys,
  T* values,
  const T* qkv,
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int sequence_length)
{
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHeadDim};
  const int row_width {(num_heads + 2 * num_kv_heads) * kHeadDim};
  // One thread per query element; K/V elements (a subset of head indices)
  // are written by the thread whose h is that group's first head.
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    const long long qkv_row {
      (static_cast<long long>(b) * sequence_length + t) * row_width};

    const long long query_out_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};
    queries[query_out_index] = qkv[qkv_row + h * kHeadDim + d];

    // The group's first head also gathers the group's shared K/V head.
    if (h % kv_group_size == 0)
    {
      const int kv_head {h / kv_group_size};
      const long long kv_out_index {
        ((static_cast<long long>(b) * num_kv_heads + kv_head) *
          sequence_length + t) * kHeadDim + d};
      keys[kv_out_index] =
        qkv[qkv_row + d_model + kv_head * kHeadDim + d];
      values[kv_out_index] =
        qkv[qkv_row + d_model + num_kv_heads * kHeadDim +
          kv_head * kHeadDim + d];
    }
  }
}

//------------------------------------------------------------------------------
/// Adjoint (= inverse) of merge_heads, for the backward pass.
///
/// merge_heads is a permutation matrix acting on the flattened tensor, and
/// the adjoint of a permutation is its inverse permutation: the gradient
/// flowing into merge_heads' output (row-major (B·T, d_model)) is scattered
/// back to merge_heads' input layout (row-major (B·NH, T, kHeadDim)) by
/// reading each element from where merge_heads would have written it. No
/// arithmetic — gradients pass through a permutation unchanged.
///
/// gradient_concatenated is row-major (B·T, d_model) — dH, the gradient
/// w.r.t. the concatenated heads. gradient_per_head is row-major
/// (B·NH, T, kHeadDim) — dO per (batch, head) slice, the layout
/// flash_attention_backward expects for its gradient_output argument.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void split_heads(
  T* gradient_per_head,
  const T* gradient_concatenated,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    const long long per_head_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};
    const long long concat_index {
      (static_cast<long long>(b) * sequence_length + t) * d_model +
        h * kHeadDim + d};

    gradient_per_head[per_head_index] = gradient_concatenated[concat_index];
  }
}

//------------------------------------------------------------------------------
/// Adjoint (= inverse) of split_qkv_heads, for the backward pass.
///
/// split_qkv_heads is a permutation from the fused GEMM's row-major
/// (B·T, 3·d_model) layout to three per-head-contiguous (B·NH, T, kHeadDim)
/// tensors; its adjoint scatters the three gradient tensors dQ, dK, dV back
/// into the fused layout, producing the gradient w.r.t. the fused GEMM's
/// output — the d(qkv) that the weight-gradient and input-gradient GEMMs of
/// the backward pass consume. Same index arithmetic as split_qkv_heads with
/// reads and writes exchanged.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void merge_qkv_heads(
  T* gradient_qkv,
  const T* gradient_queries,
  const T* gradient_keys,
  const T* gradient_values,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    const long long qkv_row {
      (static_cast<long long>(b) * sequence_length + t) * 3 * d_model};
    const long long head_offset {
      static_cast<long long>(h) * kHeadDim + d};

    const long long per_head_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};

    gradient_qkv[qkv_row + head_offset] = gradient_queries[per_head_index];
    gradient_qkv[qkv_row + d_model + head_offset] =
      gradient_keys[per_head_index];
    gradient_qkv[qkv_row + 2 * d_model + head_offset] =
      gradient_values[per_head_index];
  }
}

//------------------------------------------------------------------------------
/// Adjoint (= inverse) of split_grouped_qkv_heads, for the grouped-query
/// backward pass. Scatters dQ (per query head, B·NH slices) and the
/// group-summed dK/dV (per KV head, B·NKV slices — the output of
/// reduce_grouped_kv_gradients, NOT the per-query-head partials) back into
/// the fused row-major (B·T, (NH + 2·NKV)·kHeadDim) layout the QKV
/// linear-map weight-gradient GEMMs consume. split_grouped_qkv_heads reads
/// each fused element exactly once, so its adjoint is the pure inverse
/// gather — the group summation already happened upstream in the
/// reduction. kv_group_size = 1 is the adjoint of the ungrouped split with
/// the [Q | K | V] layout intact.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void merge_grouped_qkv_heads(
  T* gradient_qkv,
  const T* gradient_queries,
  const T* gradient_keys,
  const T* gradient_values,
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int sequence_length)
{
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHeadDim};
  const int row_width {(num_heads + 2 * num_kv_heads) * kHeadDim};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};

  for (
    long long index {
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x};
    index < total_elements;
    index += static_cast<long long>(gridDim.x) * blockDim.x)
  {
    const int d {static_cast<int>(index % kHeadDim)};
    const int t {static_cast<int>((index / kHeadDim) % sequence_length)};
    const int h {
      static_cast<int>(
        (index / (static_cast<long long>(kHeadDim) * sequence_length)) %
          num_heads)};
    const int b {
      static_cast<int>(
        index / (static_cast<long long>(kHeadDim) * sequence_length *
          num_heads))};

    const long long qkv_row {
      (static_cast<long long>(b) * sequence_length + t) * row_width};

    const long long query_in_index {
      ((static_cast<long long>(b) * num_heads + h) * sequence_length + t) *
        kHeadDim + d};
    gradient_qkv[qkv_row + h * kHeadDim + d] =
      gradient_queries[query_in_index];

    // The group's first head also scatters the group's shared dK/dV column.
    if (h % kv_group_size == 0)
    {
      const int kv_head {h / kv_group_size};
      const long long kv_in_index {
        ((static_cast<long long>(b) * num_kv_heads + kv_head) *
          sequence_length + t) * kHeadDim + d};
      gradient_qkv[qkv_row + d_model + kv_head * kHeadDim + d] =
        gradient_keys[kv_in_index];
      gradient_qkv[qkv_row + d_model + num_kv_heads * kHeadDim +
        kv_head * kHeadDim + d] = gradient_values[kv_in_index];
    }
  }
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_SPLIT_QKV_HEADS_H
