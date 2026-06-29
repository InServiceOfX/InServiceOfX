#ifndef LLM_ATTENTION_FORWARD_FLASH_ATTENTION_H
#define LLM_ATTENTION_FORWARD_FLASH_ATTENTION_H

#include <float.h>
#include "Numerics/MathFunctions.h"

namespace LLM
{
namespace AttentionForward
{

//------------------------------------------------------------------------------
/// \brief FlashAttention forward pass — IO-aware exact attention.
///
/// Avoids materializing the N×N attention matrix to HBM by computing attention
/// in SRAM tiles. HBM writes are O(Nd) rather than O(N²).
///
/// Algorithm: Dao et al. (2022) FlashAttention, Algorithm 1.
/// https://arxiv.org/abs/2205.14135
///
/// Template parameters:
///   FPType     — floating point type (float or __half)
///   kHeadDim   — head dimension d (compile-time, enables register arrays)
///   kBlockRowSize  — Br: rows of Q processed per block
///   kBlockColSize  — Bc: cols of K/V loaded per inner iteration
///
/// Grid:  (Tr, B*NH)   where Tr = ceil(N / kBlockRowSize)
/// Block: (kBlockRowSize, 1)
///
/// Shared memory per block (bytes):
///   sizeof(FPType) * (kBlockRowSize + 2*kBlockColSize) * kHeadDim
///
/// \param[out] output   Shape [B, NH, N, kHeadDim]
/// \param[in]  Q        Shape [B, NH, N, kHeadDim]
/// \param[in]  K        Shape [B, NH, N, kHeadDim]
/// \param[in]  V        Shape [B, NH, N, kHeadDim]
/// \param[in]  N        Sequence length
/// \param[in]  scale    Softmax scale factor, typically 1/sqrt(kHeadDim)
//------------------------------------------------------------------------------
template <typename FPType, int kHeadDim, int kBlockRowSize, int kBlockColSize>
__global__ void flash_attention_forward_kernel(
  FPType* __restrict__ output,
  const FPType* __restrict__ Q,
  const FPType* __restrict__ K,
  const FPType* __restrict__ V,
  const int N,
  const FPType scale)
{
  // ── Shared memory layout ─────────────────────────────────────────────────
  // Q_tile:  kBlockRowSize × kHeadDim
  // K_tile:  kBlockColSize × kHeadDim
  // V_tile:  kBlockColSize × kHeadDim
  extern __shared__ FPType shared_mem[];
  FPType* Q_tile = shared_mem;
  FPType* K_tile = Q_tile + kBlockRowSize * kHeadDim;
  FPType* V_tile = K_tile + kBlockColSize * kHeadDim;

  // ── Thread / block indexing ──────────────────────────────────────────────
  // blockIdx.y = flattened (batch, head) index
  // blockIdx.x = row-block index i (which Q tile)
  const int bh_idx = blockIdx.y;
  const int row_block = blockIdx.x;
  const int thread_row = threadIdx.x;  // row within Q_i tile

  const int row = row_block * kBlockRowSize + thread_row;

  const FPType* Q_bh = Q + bh_idx * N * kHeadDim;
  const FPType* K_bh = K + bh_idx * N * kHeadDim;
  const FPType* V_bh = V + bh_idx * N * kHeadDim;
  FPType* O_bh = output + bh_idx * N * kHeadDim;

  // ── Load Q tile into shared memory ──────────────────────────────────────
  // All threads cooperate: thread t loads row t of the Q tile.
  if (row < N)
  {
    for (int d = 0; d < kHeadDim; ++d)
    {
      Q_tile[thread_row * kHeadDim + d] = Q_bh[row * kHeadDim + d];
    }
  }
  __syncthreads();

  // ── Per-thread accumulators (in registers) ───────────────────────────────
  FPType O_row[kHeadDim];
  for (int d = 0; d < kHeadDim; ++d)
  {
    O_row[d] = static_cast<FPType>(0);
  }
  FPType running_max = -FLT_MAX;
  FPType running_sum = static_cast<FPType>(0);

  if (row >= N)
  {
    return;
  }

  const int num_col_blocks = (N + kBlockColSize - 1) / kBlockColSize;

  // ── Outer loop: iterate over K/V tiles ──────────────────────────────────
  for (int col_block = 0; col_block < num_col_blocks; ++col_block)
  {
    // ── Load K_j and V_j tiles (all threads cooperate) ────────────────────
    // Thread t loads row (t % kBlockColSize) of K_tile and V_tile.
    // Multiple passes if kBlockRowSize > kBlockColSize.
    for (int load_row = thread_row; load_row < kBlockColSize; load_row += kBlockRowSize)
    {
      const int col = col_block * kBlockColSize + load_row;
      if (col < N)
      {
        for (int d = 0; d < kHeadDim; ++d)
        {
          K_tile[load_row * kHeadDim + d] = K_bh[col * kHeadDim + d];
          V_tile[load_row * kHeadDim + d] = V_bh[col * kHeadDim + d];
        }
      }
      else
      {
        for (int d = 0; d < kHeadDim; ++d)
        {
          K_tile[load_row * kHeadDim + d] = static_cast<FPType>(0);
          V_tile[load_row * kHeadDim + d] = static_cast<FPType>(0);
        }
      }
    }
    __syncthreads();

    // ── Compute S_row = Q_i[thread_row] · K_j^T, find tile max ───────────
    FPType S_row[kBlockColSize];
    FPType tile_max = -FLT_MAX;

    for (int jj = 0; jj < kBlockColSize; ++jj)
    {
      const int col = col_block * kBlockColSize + jj;
      if (col < N)
      {
        FPType dot = static_cast<FPType>(0);
        for (int d = 0; d < kHeadDim; ++d)
        {
          dot += Q_tile[thread_row * kHeadDim + d] * K_tile[jj * kHeadDim + d];
        }
        S_row[jj] = dot * scale;
        tile_max = Numerics::MathFunctions::get_max<FPType>(tile_max, S_row[jj]);
      }
      else
      {
        // Padding columns masked to -inf so they do not contribute.
        S_row[jj] = -FLT_MAX;
      }
    }

    // ── Compute P_row = exp(S_row - tile_max), tile_sum = sum(P_row) ──────
    FPType P_row[kBlockColSize];
    FPType tile_sum = static_cast<FPType>(0);
    for (int jj = 0; jj < kBlockColSize; ++jj)
    {
      P_row[jj] = Numerics::MathFunctions::get_exponential<FPType>(
        S_row[jj] - tile_max);
      tile_sum += P_row[jj];
    }

    // ── Update running max and sum (online softmax rescaling) ─────────────
    // m_new = max(m_i, m_ij)
    // l_new = exp(m_i - m_new)*l_i + exp(m_ij - m_new)*l_ij
    const FPType new_max = Numerics::MathFunctions::get_max<FPType>(
      running_max, tile_max);
    const FPType alpha = Numerics::MathFunctions::get_exponential<FPType>(
      running_max - new_max);
    const FPType beta = Numerics::MathFunctions::get_exponential<FPType>(
      tile_max - new_max);
    const FPType new_sum = alpha * running_sum + beta * tile_sum;

    // ── Update O_row ───────────────────────────────────────────────────────
    // O_new = (l_old * exp(m_old - m_new) * O_old + exp(m_tile - m_new) * P V)
    //       / l_new
    // Maintain O unnormalized (multiply by l_new at end to avoid extra division).
    // Here we keep O in the form: O_row = (running_sum * O_row), then renorm.
    for (int d = 0; d < kHeadDim; ++d)
    {
      FPType pv = static_cast<FPType>(0);
      for (int jj = 0; jj < kBlockColSize; ++jj)
      {
        pv += P_row[jj] * V_tile[jj * kHeadDim + d];
      }
      O_row[d] = (alpha * running_sum * O_row[d] + beta * pv) / new_sum;
    }

    running_max = new_max;
    running_sum = new_sum;

    __syncthreads();
  }

  // ── Write output ──────────────────────────────────────────────────────────
  // O_row is already normalized (divided by l throughout).
  for (int d = 0; d < kHeadDim; ++d)
  {
    O_bh[row * kHeadDim + d] = O_row[d];
  }
}

//------------------------------------------------------------------------------
/// \brief Launch wrapper for flash_attention_forward_kernel with common defaults.
///
/// Computes attention for one (batch, head) stream at a time.
/// For a full multi-head multi-batch run, call this in a loop or extend the grid.
///
/// Default tile sizes: kBlockRowSize=32, kBlockColSize=32, kHeadDim=64.
/// These fit in ~24 KB shared memory (well within SM 8.6 limits of 100 KB).
//------------------------------------------------------------------------------
template <
  typename FPType = float,
  int kHeadDim = 64,
  int kBlockRowSize = 32,
  int kBlockColSize = 32>
void flash_attention_forward(
  FPType* output,
  const FPType* Q,
  const FPType* K,
  const FPType* V,
  const int batch_size,
  const int num_heads,
  const int seq_length,
  const FPType scale)
{
  const int num_row_blocks = (seq_length + kBlockRowSize - 1) / kBlockRowSize;

  // Grid covers all row blocks × all (batch, head) pairs.
  const dim3 grid(num_row_blocks, batch_size * num_heads);
  const dim3 block(kBlockRowSize, 1);

  const size_t shared_bytes = sizeof(FPType) *
    (kBlockRowSize + 2 * kBlockColSize) * kHeadDim;

  flash_attention_forward_kernel<FPType, kHeadDim, kBlockRowSize, kBlockColSize>
    <<<grid, block, shared_bytes>>>(
      output, Q, K, V, seq_length, scale);
}

} // namespace AttentionForward
} // namespace LLM

#endif // LLM_ATTENTION_FORWARD_FLASH_ATTENTION_H
