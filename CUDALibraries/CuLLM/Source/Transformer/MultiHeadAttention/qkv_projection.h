#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_QKV_PROJECTION_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_QKV_PROJECTION_H

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
/// Computes Q, K, V for every head via one fused GEMM (a genuine linear map
/// into a lower-dimensional space — d_k < d_model — not a projection in the
/// algebraic P² = P sense; see the remark on "projected" and "learned" in
/// the section on Setup: Sequences as Matrix Rows in FlashAttention.tex)
/// plus the per-head gather of split_qkv_heads.
///
/// Computes qkv := X W_qkv, row-major (B·T)×(3·d_model), then splits it into
/// per-head-contiguous queries, keys, values (see split_qkv_heads.h).
///
/// Row-major GEMM via cuBLASLt's column-major API:
/// cuBLASLt's cublasLtMatmul computes D = A·B with all matrices *column*-
/// major (verified directly against LtMatrixMultiplicationTests in
/// MoreCUDA). To get the row-major product Out(m,n) = X(m,k)·W(k,n) without
/// any physical transpose, use the standard identity
///   Out^⊤ = W^⊤ X^⊤,
/// together with the fact that a row-major (r,c) buffer, read again as
/// column-major with shape (c,r), *is* that matrix's transpose (same
/// bytes). So X's row-major (m,k) buffer is X^⊤ column-major (k,m); W's
/// row-major (k,n) buffer is W^⊤ column-major (n,k); calling the
/// column-major GEMM with A := W (as an (n,k) matrix), B := X (as a (k,m)
/// matrix) computes W^⊤X^⊤ = Out^⊤ column-major (n,m) — whose raw bytes are
/// exactly Out row-major (m,n). No data movement, only relabelling which
/// operand is "A" and which dimension is M vs. N:
///   M_cublas = n = output column count (here 3·d_model),
///   K_cublas = k = contraction dimension (here d_model),
///   N_cublas = m = row count (here num_tokens = B·T).
///
/// input is row-major (B·T, d_model) — tokens of all batch elements
/// stacked, i.e. X of Definition~(Scaled Dot-Product Attention) applied
/// per-head after the split below.
/// qkv_weights is row-major (d_model, 3·d_model): see split_qkv_heads.h for
/// the exact column layout ([W^Q | W^K | W^V], each further split by head).
/// qkv_workspace is caller-allocated row-major (B·T, 3·d_model) scratch for
/// the GEMM output, consumed and discarded by the split.
/// queries, keys, values are each row-major (B·NH, T, kHeadDim), ready for
/// flash_attention (or flash_attention_warp_cooperative).
///
/// kHeadDim is a compile-time constant, matching every Attention/ kernel.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
bool qkv_projection(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* queries,
  T* keys,
  T* values,
  T* qkv_workspace,
  const T* input,
  const T* qkv_weights,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHeadDim};
  const int num_tokens {batch_size * sequence_length};

  cuBLASWrappers::MatrixMultiplication::Setup<T> setup(
    3 * d_model, num_tokens, d_model);
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
    qkv_weights,
    input,
    nullptr,
    qkv_workspace))
  {
    return false;
  }

  constexpr int kThreadsPerBlock {256};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};
  const int number_of_blocks {static_cast<int>(
    (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  split_qkv_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    queries,
    keys,
    values,
    qkv_workspace,
    batch_size,
    num_heads,
    sequence_length);

  return true;
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_QKV_PROJECTION_H
