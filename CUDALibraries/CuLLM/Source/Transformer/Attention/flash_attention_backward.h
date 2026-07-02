#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_BACKWARD_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_BACKWARD_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// FlashAttention backward pass (see the section on The FlashAttention
/// Backward Pass in FlashAttention.tex).
///
/// The forward pass stored only O and the per-row logsumexp L_i = m + ln ℓ;
/// S and P were discarded. The backward pass recomputes weight tiles in
/// SRAM as P_ij = exp(S_ij − L_i) (see the section on Recomputation and the
/// Logsumexp Statistic) and accumulates the gradient formulas (see the
/// section on Gradients of Scaled Dot-Product Attention):
///
///   D_i  = ⟨dO_i, O_i⟩                       (the row-dot identity)
///   dP_ij = ⟨dO_i, v_j⟩
///   dS_ij = P_ij (dP_ij − D_i)               (softmax Jacobian, row-wise)
///   dQ_i = Σ_j dS_ij k_j / √d_k
///   dK_j = Σ_i dS_ij q_i / √d_k
///   dV_j = Σ_i P_ij dO_i
///
/// Unlike the forward pass there is no merge monoid here: L_i is already a
/// global statistic, so every sum above is linear and decomposes over tiles
/// exactly — tile accumulation is plain addition.
///
/// Organization: three kernels, each output block having a single writer
/// (no atomics):
///   1. attention_backward_row_dots      — D = rowsum(dO ⊙ O)
///   2. ..._query_gradient (Pass 1)      — warp per query row, dQ
///   3. ..._key_value_gradient (Pass 2)  — warp per key row, dK and dV
/// Passes 1 and 2 reuse the forward warp-cooperative mapping: lane ℓ of a
/// warp handles one key (Pass 1) or one query (Pass 2) of a 32-wide tile,
/// per-key/per-query scalars are shuffle-broadcast, and each lane owns the
/// interleaved fragment c = f·32 + lane of its gradient row.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// D = rowsum(dO ⊙ O) ∈ R^n: one warp per row, lane partial sums folded by
/// a warp tree reduction. By the row-dot identity D_i = Σ_k P_ik dP_ik, this
/// is the only global use of dP — computed in Θ(nd) without materializing
/// the n×n matrix dP.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock>
__global__ void attention_backward_row_dots(
  T* row_dots,
  const T* gradient_output,
  const T* output,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;
  constexpr int WARP_SIZE {32};

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  cg::thread_block_tile<WARP_SIZE> warp {cg::tiled_partition<WARP_SIZE>(block)};

  const int slice_matrix_offset {
    static_cast<int>(blockIdx.y) * sequence_length * kHeadDim};
  gradient_output += slice_matrix_offset;
  output += slice_matrix_offset;
  row_dots += static_cast<int>(blockIdx.y) * sequence_length;

  const int row {
    static_cast<int>(blockIdx.x) * kWarpsPerBlock +
      static_cast<int>(warp.meta_group_rank())};
  if (row >= sequence_length)
  {
    return;
  }

  AccT partial {0};
  for (
    int d {static_cast<int>(warp.thread_rank())};
    d < kHeadDim;
    d += WARP_SIZE)
  {
    partial += static_cast<AccT>(gradient_output[row * kHeadDim + d]) *
      static_cast<AccT>(output[row * kHeadDim + d]);
  }
  const AccT total {cg::reduce(warp, partial, cg::plus<AccT>())};

  if (warp.thread_rank() == 0)
  {
    row_dots[row] = static_cast<T>(total);
  }
}

//------------------------------------------------------------------------------
/// Pass 1: dQ_i = (Σ_j dS_ij k_j) / √d_k, one warp per query row.
///
/// Per K/V tile (32 keys, one per lane), lane ℓ recomputes its key's weight
/// and score gradient
///   p_ℓ  = exp(q_i · k_ℓ / √d_k − L_i),
///   dp_ℓ = ⟨dO_i, v_ℓ⟩,
///   ds_ℓ = p_ℓ (dp_ℓ − D_i),
/// then all lanes accumulate dQ fragments with ds_j shuffle-broadcast:
///   dq[f·32 + t] += Σ_j ds_j · K_j[f·32 + t].
/// The 1/√d_k of dQ = dS K/√d_k is applied once in the epilogue.
///
/// Masked keys (out of range, or j > i under kCausal) take p_ℓ = 0, hence
/// ds_ℓ = 0 — they contribute nothing, matching the causal remark in the
/// gradients section (dS_ij = 0 wherever P_ij = 0). Under kCausal the tile
/// loop also stops after the block's last query row, as in the forward.
///
/// Q and dO rows are read at warp-uniform addresses (broadcast, unpadded);
/// K and V rows are both lane-walked (dot products) and lane-striped
/// (fragment accumulation), so both are padded by one column.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
__global__ void flash_attention_backward_query_gradient(
  T* gradient_queries,
  const T* queries,
  const T* keys,
  const T* values,
  const T* gradient_output,
  const T* logsumexp,
  const T* row_dots,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;
  constexpr int WARP_SIZE {32};
  constexpr int kTileColumns {WARP_SIZE};
  constexpr int kFragment {kHeadDim / WARP_SIZE};
  static_assert(kHeadDim % WARP_SIZE == 0);

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  cg::thread_block_tile<WARP_SIZE> warp {cg::tiled_partition<WARP_SIZE>(block)};
  const int warp_rank {static_cast<int>(warp.meta_group_rank())};
  const int lane {static_cast<int>(warp.thread_rank())};

  const int slice_matrix_offset {
    static_cast<int>(blockIdx.y) * sequence_length * kHeadDim};
  const int slice_row_offset {
    static_cast<int>(blockIdx.y) * sequence_length};
  gradient_queries += slice_matrix_offset;
  queries += slice_matrix_offset;
  keys += slice_matrix_offset;
  values += slice_matrix_offset;
  gradient_output += slice_matrix_offset;
  logsumexp += slice_row_offset;
  row_dots += slice_row_offset;

  const int query_index {
    static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_rank};

  __shared__ AccT shared_queries[kWarpsPerBlock][kHeadDim];
  __shared__ AccT shared_gradient_output[kWarpsPerBlock][kHeadDim];
  __shared__ AccT shared_keys[kTileColumns][kHeadDim + 1];
  __shared__ AccT shared_values[kTileColumns][kHeadDim + 1];

  // Load the block's Q and dO rows once (coalesced cooperative loads).
  for (
    int index {static_cast<int>(threadIdx.x)};
    index < kWarpsPerBlock * kHeadDim;
    index += static_cast<int>(blockDim.x))
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int global_row {
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + row};
    const bool in_range {global_row < sequence_length};
    shared_queries[row][d] = in_range ?
      static_cast<AccT>(queries[global_row * kHeadDim + d]) : AccT{0};
    shared_gradient_output[row][d] = in_range ?
      static_cast<AccT>(gradient_output[global_row * kHeadDim + d]) : AccT{0};
  }

  const AccT scale {
    AccT{1} /
      Numerics::MathFunctions::get_sqrt<AccT>(static_cast<AccT>(kHeadDim))};

  // Per-row statistics from the forward pass, warp-uniform.
  const AccT row_logsumexp {(query_index < sequence_length) ?
    static_cast<AccT>(logsumexp[query_index]) : AccT{0}};
  const AccT row_dot {(query_index < sequence_length) ?
    static_cast<AccT>(row_dots[query_index]) : AccT{0}};

  AccT fragment[kFragment];
  #pragma unroll
  for (int f {0}; f < kFragment; ++f)
  {
    fragment[f] = AccT{0};
  }

  int number_of_tiles {
    (sequence_length + kTileColumns - 1) / kTileColumns};
  if (kCausal)
  {
    const int last_query_in_block {
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + kWarpsPerBlock - 1};
    const int last_needed_tile {last_query_in_block / kTileColumns};
    number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
      last_needed_tile + 1 : number_of_tiles;
  }

  for (int tile {0}; tile < number_of_tiles; ++tile)
  {
    const int tile_start {tile * kTileColumns};

    __syncthreads();
    for (
      int index {static_cast<int>(threadIdx.x)};
      index < kTileColumns * kHeadDim;
      index += static_cast<int>(blockDim.x))
    {
      const int row {index / kHeadDim};
      const int d {index % kHeadDim};
      const int global_row {tile_start + row};
      const bool in_range {global_row < sequence_length};
      shared_keys[row][d] = in_range ?
        static_cast<AccT>(keys[global_row * kHeadDim + d]) : AccT{0};
      shared_values[row][d] = in_range ?
        static_cast<AccT>(values[global_row * kHeadDim + d]) : AccT{0};
    }
    __syncthreads();

    if (query_index >= sequence_length)
    {
      continue;
    }

    const int key_index {tile_start + lane};
    const bool masked {
      key_index >= sequence_length ||
      (kCausal && key_index > query_index)};

    // ds_ℓ = P_ij (dP_ij − D_i) for this lane's key; 0 where P_ij = 0.
    AccT score_gradient {0};
    if (!masked)
    {
      AccT score {0};
      AccT weight_gradient {0};
      #pragma unroll
      for (int d {0}; d < kHeadDim; ++d)
      {
        score += shared_queries[warp_rank][d] * shared_keys[lane][d];
        weight_gradient +=
          shared_gradient_output[warp_rank][d] * shared_values[lane][d];
      }
      const AccT weight {Numerics::MathFunctions::get_exponential<AccT>(
        score * scale - row_logsumexp)};
      score_gradient = weight * (weight_gradient - row_dot);
    }

    // dq[f·32 + t] += Σ_j ds_j · K_j[f·32 + t], ds_j broadcast from lane j.
    #pragma unroll
    for (int j {0}; j < kTileColumns; ++j)
    {
      const AccT ds_j {warp.shfl(score_gradient, j)};
      #pragma unroll
      for (int f {0}; f < kFragment; ++f)
      {
        fragment[f] += ds_j * shared_keys[j][f * WARP_SIZE + lane];
      }
    }
  }

  // Epilogue: dQ_i = (Σ_j ds_j k_j) · (1/√d_k).
  if (query_index < sequence_length)
  {
    #pragma unroll
    for (int f {0}; f < kFragment; ++f)
    {
      gradient_queries[query_index * kHeadDim + f * WARP_SIZE + lane] =
        static_cast<T>(fragment[f] * scale);
    }
  }
}

//------------------------------------------------------------------------------
/// Pass 2: dK_j = (Σ_i dS_ij q_i) / √d_k and dV_j = Σ_i P_ij dO_i, one warp
/// per *key* row — the transpose of Pass 1's mapping. Per query tile
/// (32 queries, one per lane), lane i recomputes p_i and ds_i for its query
/// against this warp's key, then the fragments accumulate with shuffle
/// broadcasts:
///   dv[f·32 + t] += Σ_i p_i  · dO_i[f·32 + t],
///   dk[f·32 + t] += Σ_i ds_i · Q_i [f·32 + t].
///
/// Causal masking blocks queries i < j (position i attends only to j ≤ i),
/// and the tile loop *starts* at the block's first key row instead of
/// stopping early — the mirror image of Pass 1, again skipping about half
/// the tiles. K and V rows of the block are warp-uniform (broadcast,
/// unpadded); Q and dO tile rows are lane-walked and lane-striped (padded).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
__global__ void flash_attention_backward_key_value_gradient(
  T* gradient_keys,
  T* gradient_values,
  const T* queries,
  const T* keys,
  const T* values,
  const T* gradient_output,
  const T* logsumexp,
  const T* row_dots,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;
  constexpr int WARP_SIZE {32};
  constexpr int kTileRows {WARP_SIZE};
  constexpr int kFragment {kHeadDim / WARP_SIZE};
  static_assert(kHeadDim % WARP_SIZE == 0);

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  cg::thread_block_tile<WARP_SIZE> warp {cg::tiled_partition<WARP_SIZE>(block)};
  const int warp_rank {static_cast<int>(warp.meta_group_rank())};
  const int lane {static_cast<int>(warp.thread_rank())};

  const int slice_matrix_offset {
    static_cast<int>(blockIdx.y) * sequence_length * kHeadDim};
  const int slice_row_offset {
    static_cast<int>(blockIdx.y) * sequence_length};
  gradient_keys += slice_matrix_offset;
  gradient_values += slice_matrix_offset;
  queries += slice_matrix_offset;
  keys += slice_matrix_offset;
  values += slice_matrix_offset;
  gradient_output += slice_matrix_offset;
  logsumexp += slice_row_offset;
  row_dots += slice_row_offset;

  const int key_index {
    static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_rank};

  __shared__ AccT shared_keys[kWarpsPerBlock][kHeadDim];
  __shared__ AccT shared_values[kWarpsPerBlock][kHeadDim];
  __shared__ AccT shared_queries[kTileRows][kHeadDim + 1];
  __shared__ AccT shared_gradient_output[kTileRows][kHeadDim + 1];

  // Load the block's K and V rows once (coalesced cooperative loads).
  for (
    int index {static_cast<int>(threadIdx.x)};
    index < kWarpsPerBlock * kHeadDim;
    index += static_cast<int>(blockDim.x))
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int global_row {
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + row};
    const bool in_range {global_row < sequence_length};
    shared_keys[row][d] = in_range ?
      static_cast<AccT>(keys[global_row * kHeadDim + d]) : AccT{0};
    shared_values[row][d] = in_range ?
      static_cast<AccT>(values[global_row * kHeadDim + d]) : AccT{0};
  }

  const AccT scale {
    AccT{1} /
      Numerics::MathFunctions::get_sqrt<AccT>(static_cast<AccT>(kHeadDim))};

  AccT key_fragment[kFragment];
  AccT value_fragment[kFragment];
  #pragma unroll
  for (int f {0}; f < kFragment; ++f)
  {
    key_fragment[f] = AccT{0};
    value_fragment[f] = AccT{0};
  }

  const int number_of_tiles {
    (sequence_length + kTileRows - 1) / kTileRows};
  int first_tile {0};
  if (kCausal)
  {
    // Query tiles wholly before the block's first key row are masked for
    // every key of this block (P_ij = 0 for i < j). Uniform per block.
    first_tile =
      (static_cast<int>(blockIdx.x) * kWarpsPerBlock) / kTileRows;
  }

  for (int tile {first_tile}; tile < number_of_tiles; ++tile)
  {
    const int tile_start {tile * kTileRows};

    __syncthreads();
    for (
      int index {static_cast<int>(threadIdx.x)};
      index < kTileRows * kHeadDim;
      index += static_cast<int>(blockDim.x))
    {
      const int row {index / kHeadDim};
      const int d {index % kHeadDim};
      const int global_row {tile_start + row};
      const bool in_range {global_row < sequence_length};
      shared_queries[row][d] = in_range ?
        static_cast<AccT>(queries[global_row * kHeadDim + d]) : AccT{0};
      shared_gradient_output[row][d] = in_range ?
        static_cast<AccT>(gradient_output[global_row * kHeadDim + d]) :
        AccT{0};
    }
    __syncthreads();

    if (key_index >= sequence_length)
    {
      continue;
    }

    const int query_index {tile_start + lane};
    const bool masked {
      query_index >= sequence_length ||
      (kCausal && query_index < key_index)};

    // p_i = P_ij and ds_i = dS_ij for this lane's query against key j.
    AccT weight {0};
    AccT score_gradient {0};
    if (!masked)
    {
      AccT score {0};
      AccT weight_gradient {0};
      #pragma unroll
      for (int d {0}; d < kHeadDim; ++d)
      {
        score += shared_queries[lane][d] * shared_keys[warp_rank][d];
        weight_gradient +=
          shared_gradient_output[lane][d] * shared_values[warp_rank][d];
      }
      weight = Numerics::MathFunctions::get_exponential<AccT>(
        score * scale - static_cast<AccT>(logsumexp[query_index]));
      score_gradient =
        weight * (weight_gradient - static_cast<AccT>(row_dots[query_index]));
    }

    #pragma unroll
    for (int i {0}; i < kTileRows; ++i)
    {
      const AccT p_i {warp.shfl(weight, i)};
      const AccT ds_i {warp.shfl(score_gradient, i)};
      #pragma unroll
      for (int f {0}; f < kFragment; ++f)
      {
        value_fragment[f] +=
          p_i * shared_gradient_output[i][f * WARP_SIZE + lane];
        key_fragment[f] += ds_i * shared_queries[i][f * WARP_SIZE + lane];
      }
    }
  }

  // Epilogue: dK_j = (Σ_i ds_i q_i) · (1/√d_k), dV_j = Σ_i p_i dO_i.
  if (key_index < sequence_length)
  {
    #pragma unroll
    for (int f {0}; f < kFragment; ++f)
    {
      gradient_keys[key_index * kHeadDim + f * WARP_SIZE + lane] =
        static_cast<T>(key_fragment[f] * scale);
      gradient_values[key_index * kHeadDim + f * WARP_SIZE + lane] =
        static_cast<T>(value_fragment[f]);
    }
  }
}

//------------------------------------------------------------------------------
/// Host-side launcher for the full backward pass: D preprocess, then the
/// two gradient passes. row_dots_workspace is a caller-allocated device
/// buffer of number_of_batch_heads · sequence_length elements. logsumexp
/// must come from the forward pass (flash_attention_warp_cooperative with a
/// non-null logsumexp argument).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
void flash_attention_backward(
  T* gradient_queries,
  T* gradient_keys,
  T* gradient_values,
  T* row_dots_workspace,
  const T* queries,
  const T* keys,
  const T* values,
  const T* output,
  const T* gradient_output,
  const T* logsumexp,
  const int sequence_length,
  const int number_of_batch_heads = 1)
{
  constexpr int WARP_SIZE {32};
  const int number_of_row_blocks {
    (sequence_length + kWarpsPerBlock - 1) / kWarpsPerBlock};
  const dim3 grid {
    static_cast<unsigned int>(number_of_row_blocks),
    static_cast<unsigned int>(number_of_batch_heads)};
  const int block_size {kWarpsPerBlock * WARP_SIZE};

  attention_backward_row_dots<T, kHeadDim, kWarpsPerBlock>
    <<<grid, block_size>>>(
      row_dots_workspace,
      gradient_output,
      output,
      sequence_length);

  flash_attention_backward_query_gradient<
    T,
    kHeadDim,
    kWarpsPerBlock,
    kCausal><<<grid, block_size>>>(
      gradient_queries,
      queries,
      keys,
      values,
      gradient_output,
      logsumexp,
      row_dots_workspace,
      sequence_length);

  flash_attention_backward_key_value_gradient<
    T,
    kHeadDim,
    kWarpsPerBlock,
    kCausal><<<grid, block_size>>>(
      gradient_keys,
      gradient_values,
      queries,
      keys,
      values,
      gradient_output,
      logsumexp,
      row_dots_workspace,
      sequence_length);
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_BACKWARD_H
