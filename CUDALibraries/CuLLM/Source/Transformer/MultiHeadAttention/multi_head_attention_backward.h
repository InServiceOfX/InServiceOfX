#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_BACKWARD_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_BACKWARD_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "StreamManagement/Stream.h"
#include "Transformer/Attention/flash_attention_backward.h"
#include "Transformer/MultiHeadAttention/split_qkv_heads.h"

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Backward pass of a right-multiplication linear map Y = X W (see the
/// section on Multi-Head Attention in FlashAttention.tex for the forward
/// maps): given dY, computes
///
///   dX = dY W^⊤          (gradient w.r.t. the input),
///   dW = X^⊤ dY          (gradient w.r.t. the learned weights),
///
/// each as one cuBLASLt GEMM using the same row-major-via-column-major
/// relabelling as qkv_linear_maps.h, extended with cuBLASLt's transpose
/// descriptor attributes (Setup::setup's is_transpose_on_A/B):
///
/// For dX (row-major (m,k) = dY(m,n) · W^⊤(n,k)): column-major it is
///   dX^⊤ = W · dY^⊤.
/// W's row-major (k,n) buffer read column-major is W^⊤ (n,k), so requesting
/// transpose-on-A recovers W as op(A); dY's row-major buffer read
/// column-major is already dY^⊤. Hence M_cublas = k, N_cublas = m,
/// K_cublas = n, A = W (transposed), B = dY.
///
/// For dW (row-major (k,n) = X^⊤(k,m) · dY(m,n)): column-major it is
///   dW^⊤ = dY^⊤ · X.
/// dY's buffer read column-major is dY^⊤ (n,m); X's row-major buffer read
/// column-major is X^⊤ (k,m), so requesting transpose-on-B recovers X as
/// op(B). Hence M_cublas = n, N_cublas = k, K_cublas = m, A = dY,
/// B = X (transposed).
//------------------------------------------------------------------------------
template <typename T>
bool linear_map_backward(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* gradient_input,
  T* gradient_weight_matrix,
  const T* gradient_output,
  const T* input,
  const T* weight_matrix,
  const int input_rows,
  const int input_columns,
  const int output_columns)
{
  const int m {input_rows};
  const int k {input_columns};
  const int n {output_columns};

  // dX = dY W^⊤: M = k, N = m, K = n, transpose on A.
  {
    cuBLASWrappers::MatrixMultiplication::Setup<T> setup(k, m, n);
    if (!setup.setup(
      handle,
      /* is_transpose_on_A = */ true,
      /* is_transpose_on_B = */ false))
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
      weight_matrix,
      gradient_output,
      nullptr,
      gradient_input))
    {
      return false;
    }
  }

  // dW = X^⊤ dY: M = n, N = k, K = m, transpose on B.
  {
    cuBLASWrappers::MatrixMultiplication::Setup<T> setup(n, k, m);
    if (!setup.setup(
      handle,
      /* is_transpose_on_A = */ false,
      /* is_transpose_on_B = */ true))
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
      gradient_output,
      input,
      nullptr,
      gradient_weight_matrix))
    {
      return false;
    }
  }

  return true;
}

//------------------------------------------------------------------------------
/// Multi-head attention backward pass: given the gradient dY of a scalar
/// loss w.r.t. the forward output Y = MHA(X), computes the gradients w.r.t.
/// the input X and both learned weight matrices W_qkv = [W^Q | W^K | W^V]
/// and W^O. The chain runs each forward stage's adjoint in reverse order:
///
///   1. merge_heads          — recompute H = [head_1 | ... | head_h] from
///                             the saved per-head attention output (the
///                             forward discarded its concat scratch).
///   2. linear_map_backward  — dW^O = H^⊤ dY and dH = dY W^{O⊤}.
///   3. split_heads          — adjoint of merge_heads: dH → per-head dO.
///   4. flash_attention_backward — dQ, dK, dV per (batch, head) slice via
///                             logsumexp recomputation (see the section on
///                             The FlashAttention Backward Pass); the
///                             (B·NH)-way grid batching is the same
///                             blockIdx.y slicing as the forward.
///   5. merge_qkv_heads      — adjoint of split_qkv_heads: dQ/dK/dV →
///                             d(qkv) in the fused (B·T, 3·d_model) layout.
///   6. linear_map_backward  — dW_qkv = X^⊤ d(qkv) and dX = d(qkv) W_qkv^⊤.
///
/// Required forward activations (all saved by multi_head_attention when its
/// logsumexp argument is non-null): input X, queries/keys/values
/// (per-head), attention_output (per-head O), logsumexp (B·NH, T).
///
/// Caller-allocated outputs:
///   gradient_input          — (B·T, d_model), dX
///   gradient_qkv_weights    — (d_model, 3·d_model), dW_qkv
///   gradient_output_weights — (d_model, d_model), dW^O
///
/// Caller-allocated workspaces:
///   concat_workspace          — (B·T, d_model), recomputed H
///   gradient_concat_workspace — (B·T, d_model), dH
///   gradient_attention_output — (B·NH, T, kHeadDim), per-head dO
///   gradient_queries/keys/values — (B·NH, T, kHeadDim) each
///   gradient_qkv_workspace    — (B·T, 3·d_model), d(qkv)
///   row_dots_workspace        — (B·NH, T), flash_attention_backward's D
///
/// Template parameters mirror multi_head_attention; kCausal must match the
/// forward call, since the recomputed weights P_ij = exp(S_ij − L_i) are
/// only valid under the same masking.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
bool multi_head_attention_backward(
  cuBLASWrappers::LibraryContextHandle& handle,
  StreamManagement::Stream& stream,
  T* gradient_input,
  T* gradient_qkv_weights,
  T* gradient_output_weights,
  T* concat_workspace,
  T* gradient_concat_workspace,
  T* gradient_attention_output,
  T* gradient_queries,
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
  const int sequence_length)
{
  static_assert(
    kHeadDim % 32 == 0,
    "multi_head_attention_backward requires kHeadDim to be a multiple of 32 "
    "(warp size), matching the forward pass constraint.");

  const int d_model {num_heads * kHeadDim};
  const int num_tokens {batch_size * sequence_length};

  constexpr int kThreadsPerBlock {256};
  const long long total_elements {
    static_cast<long long>(batch_size) * num_heads * sequence_length *
      kHeadDim};
  const int number_of_blocks {static_cast<int>(
    (total_elements + kThreadsPerBlock - 1) / kThreadsPerBlock)};

  // 1. Recompute H (the forward's concat scratch was discarded).
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

  // 3. dH → per-head dO (adjoint of merge_heads).
  split_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    gradient_attention_output,
    gradient_concat_workspace,
    batch_size,
    num_heads,
    sequence_length);

  // 4. dQ, dK, dV for every (batch, head) slice at once.
  Attention::flash_attention_backward<T, kHeadDim, kWarpsPerBlock, kCausal>(
    gradient_queries,
    gradient_keys,
    gradient_values,
    row_dots_workspace,
    queries,
    keys,
    values,
    attention_output,
    gradient_attention_output,
    logsumexp,
    sequence_length,
    batch_size * num_heads);

  // 5. dQ/dK/dV → d(qkv) in the fused layout (adjoint of split_qkv_heads).
  merge_qkv_heads<T, kHeadDim><<<number_of_blocks, kThreadsPerBlock>>>(
    gradient_qkv_workspace,
    gradient_queries,
    gradient_keys,
    gradient_values,
    batch_size,
    num_heads,
    sequence_length);

  // 6. dW_qkv = X^⊤ d(qkv), dX = d(qkv) W_qkv^⊤.
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
    3 * d_model);
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_MULTI_HEAD_ATTENTION_BACKWARD_H
