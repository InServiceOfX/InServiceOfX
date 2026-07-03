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

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_SPLIT_QKV_HEADS_H
