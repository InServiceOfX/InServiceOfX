#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_OUTPUT_PROJECTION_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_OUTPUT_PROJECTION_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/split_qkv_heads.h"

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Computes the output projection of multi-head attention (see the section
/// on Multi-Head Attention in FlashAttention.tex):
///
///   MHA(y) := [head_1 | ... | head_h] W^O,
///
/// i.e. merge_heads (the concatenation) followed by one GEMM against
/// W^O ∈ R^{d_model × d_model}.
///
/// Row-major GEMM via cuBLASLt's column-major API: same relabelling as
/// qkv_projection.h (see that file's header comment for the derivation),
/// with M_cublas = n = d_model, K_cublas = k = d_model,
/// N_cublas = m = num_tokens = B·T.
///
/// per_head_attention_output is row-major (B·NH, T, kHeadDim) — the layout
/// flash_attention writes.
/// concat_workspace is caller-allocated row-major (B·T, d_model) scratch
/// for merge_heads' output, consumed and discarded by the GEMM.
/// output_weights is row-major (d_model, d_model), W^O.
/// output is row-major (B·T, d_model), the final MHA(y).
///
/// kHeadDim is a compile-time constant, matching every Attention/ kernel.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
bool output_projection(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* output,
  T* concat_workspace,
  const T* per_head_attention_output,
  const T* output_weights,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const int num_tokens {batch_size * sequence_length};

  constexpr int kThreadsPerBlock {256};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};
  const int number_of_blocks {static_cast<int>(
    (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  merge_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    concat_workspace,
    per_head_attention_output,
    batch_size,
    num_heads,
    sequence_length);

  cuBLASWrappers::MatrixMultiplication::Setup<T> setup(
    d_model, num_tokens, d_model);
  if (!setup.setup(handle))
  {
    return false;
  }

  cuBLASWrappers::MatrixMultiplication::LtMatrixMultiplication<T> matmul{};
  return matmul(
    handle,
    setup.descriptor_,
    setup.layouts_,
    setup.heuristic_,
    stream,
    setup.workspace_,
    output_weights,
    concat_workspace,
    nullptr,
    output);
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_OUTPUT_PROJECTION_H
