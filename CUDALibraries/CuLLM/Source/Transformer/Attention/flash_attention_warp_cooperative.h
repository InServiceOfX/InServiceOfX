#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_WARP_COOPERATIVE_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_WARP_COOPERATIVE_H

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// FlashAttention forward pass with one *warp* per query row (see
/// flash_attention_forward.h for the algorithm; this kernel changes only the
/// execution mapping, not the math).
///
/// Why: the one-thread-per-row kernel is latency-bound — each thread
/// serially computes B_c·d_k multiply-adds per tile, and a Q block of B_r
/// rows yields only B_r threads of parallelism. Assigning a warp to each
/// row multiplies the parallelism per row by 32 and keeps the GPU saturated
/// even at small n.
///
/// The accumulator (m, ℓ, õ) is *distributed across the warp*:
///   - m and ℓ are replicated in every lane (cg::reduce returns the reduced
///     value to all lanes, so the replicas stay equal by construction);
///   - õ ∈ R^{d_v} is sharded lane-wise: lane t owns the interleaved
///     components c = f·32 + t for f = 0, ..., d_v/32 − 1. Interleaving
///     makes the V-tile reads and the final O writes coalesced (consecutive
///     lanes touch consecutive addresses).
/// This is the same monoid element as AttentionAccumulator — only its
/// storage is spread over 32 lanes; the merge is executed cooperatively.
///
/// Work distribution per K/V tile (tile width = warp size, one key per
/// lane): lane ℓ computes the single dot product s_ℓ = q_i · k_ℓ / √d_k, so
/// the B_c dot products of one row of S_ij run in parallel across lanes
/// instead of serially in one thread. The row max m_ij and row sum ℓ_ij are
/// warp tree reductions (O(log 32) shuffles, valid because max and + are
/// commutative and associative). The õ update needs every lane to see every
/// p_j, one shuffle broadcast per key:
///   õ[f·32 + t] += Σ_j p_j · V_j[f·32 + t].
///
/// Shared-memory layout: Q rows are read at the same (row, d) by all lanes
/// of a warp — a broadcast, no padding needed. K rows are walked one row
/// per lane, so K is padded by one column (stride 33 mod 32 = 1 puts
/// distinct rows in distinct banks). V is read lane-striped (consecutive
/// lanes, consecutive columns) — conflict-free unpadded.
///
/// If logsumexp is non-null, lane 0 of each warp additionally writes the
/// row statistic L_i = m_i + ln(ℓ_i), laid out (batch·heads, n). This is
/// the single per-row scalar the FlashAttention backward pass stores in
/// place of S and P: the attention weights are recomputable tile-wise as
/// P_ij = exp(S_ij − L_i), since exp(s − m)/ℓ = exp(s − m − ln ℓ).
///
/// Causal masking, batch/head grid dimension, −∞ padding of out-of-range
/// keys, and the fused tile merge are identical to flash_attention_forward.
///
/// kHeadDim must be a multiple of the warp size (32) for the lane sharding.
/// kWarpsPerBlock = query rows per block (blockDim.x = 32·kWarpsPerBlock).
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
__global__ void flash_attention_forward_warp_cooperative(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;

  constexpr int WARP_SIZE {32};
  // One key per lane: the K/V tile is exactly one warp wide.
  constexpr int kTileColumns {WARP_SIZE};
  // Components of õ owned by each lane.
  constexpr int kFragment {kHeadDim / WARP_SIZE};
  static_assert(
    kHeadDim % WARP_SIZE == 0,
    "lane sharding requires kHeadDim to be a multiple of the warp size");
  static_assert(kWarpsPerBlock > 0);

  namespace cg = cooperative_groups;
  cg::thread_block block {cg::this_thread_block()};
  cg::thread_block_tile<WARP_SIZE> warp {cg::tiled_partition<WARP_SIZE>(block)};
  const int warp_rank {static_cast<int>(warp.meta_group_rank())};
  const int lane {static_cast<int>(warp.thread_rank())};

  // Each (batch, head) slice is an independent attention problem.
  const int slice_offset {
    static_cast<int>(blockIdx.y) * sequence_length * kHeadDim};
  output += slice_offset;
  queries += slice_offset;
  keys += slice_offset;
  values += slice_offset;
  if (logsumexp != nullptr)
  {
    logsumexp += static_cast<int>(blockIdx.y) * sequence_length;
  }

  const int query_index {
    static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_rank};

  __shared__ AccT shared_queries[kWarpsPerBlock][kHeadDim];
  __shared__ AccT shared_keys[kTileColumns][kHeadDim + 1];
  __shared__ AccT shared_values[kTileColumns][kHeadDim];

  // Load the block's Q rows once (coalesced cooperative load).
  for (
    int index {static_cast<int>(threadIdx.x)};
    index < kWarpsPerBlock * kHeadDim;
    index += static_cast<int>(blockDim.x))
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int global_row {
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + row};
    shared_queries[row][d] = (global_row < sequence_length) ?
      static_cast<AccT>(queries[global_row * kHeadDim + d]) : AccT{0};
  }

  const AccT scale {
    AccT{1} /
      Numerics::MathFunctions::get_sqrt<AccT>(static_cast<AccT>(kHeadDim))};
  const AccT negative_infinity {
    -Numerics::Constants::get_infinity<AccT>()};

  // Distributed accumulator, initialized to the monoid identity (−∞, 0, 0):
  // (m, ℓ) replicated per lane, õ sharded as fragment[f] = õ[f·32 + lane].
  AccT max_value {negative_infinity};
  AccT sum {0};
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
    // Skip tiles past the block's last query row (uniform per block, so the
    // __syncthreads() below stay aligned).
    const int last_query_in_block {
      static_cast<int>(blockIdx.x) * kWarpsPerBlock + kWarpsPerBlock - 1};
    const int last_needed_tile {last_query_in_block / kTileColumns};
    number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
      last_needed_tile + 1 : number_of_tiles;
  }

  for (int tile {0}; tile < number_of_tiles; ++tile)
  {
    const int tile_start {tile * kTileColumns};
    const int tile_size {
      (sequence_length - tile_start < kTileColumns) ?
        sequence_length - tile_start : kTileColumns};

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

    // Lane ℓ computes s_ℓ = q_i · k_ℓ / √d_k for its key; out-of-range and
    // causally masked keys get −∞ (the empty-element convention).
    const bool masked {
      lane >= tile_size ||
      (kCausal && tile_start + lane > query_index)};
    AccT score {negative_infinity};
    if (!masked)
    {
      AccT dot {0};
      #pragma unroll
      for (int d {0}; d < kHeadDim; ++d)
      {
        dot += shared_queries[warp_rank][d] * shared_keys[lane][d];
      }
      score = dot * scale;
    }

    // m_ij = rowmax(S_ij): warp tree reduction, result in every lane.
    const AccT tile_max {
      cg::reduce(warp, score, cg::greater<AccT>())};
    const AccT new_max {Numerics::MathFunctions::get_max<AccT>(
      max_value,
      tile_max)};
    // e^{−∞ − m_new} = 0 on the first tile: the identity rescales away.
    const AccT rescale {Numerics::MathFunctions::get_exponential<AccT>(
      max_value - new_max)};
    max_value = new_max;

    // p_ℓ = e^{s_ℓ − m_new} (this lane's key's unnormalized weight; 0 for
    // masked keys); ℓ_ij = row sum of p is another warp reduction.
    const AccT p {Numerics::MathFunctions::get_exponential<AccT>(
      score - new_max)};
    sum = sum * rescale + cg::reduce(warp, p, cg::plus<AccT>());

    // õ update: rescale this lane's shard once, then accumulate every
    // key's contribution — p_j lives in lane j, one shuffle broadcast each.
    #pragma unroll
    for (int f {0}; f < kFragment; ++f)
    {
      fragment[f] *= rescale;
    }
    #pragma unroll
    for (int j {0}; j < kTileColumns; ++j)
    {
      const AccT p_j {warp.shfl(p, j)};
      #pragma unroll
      for (int f {0}; f < kFragment; ++f)
      {
        fragment[f] += p_j * shared_values[j][f * WARP_SIZE + lane];
      }
    }
  }

  // Epilogue: O_i = õ/ℓ (each lane writes its shard, coalesced) and
  // optionally the logsumexp L_i = m_i + ln(ℓ_i) for the backward pass.
  if (query_index < sequence_length)
  {
    const AccT normalization {AccT{1} / sum};
    #pragma unroll
    for (int f {0}; f < kFragment; ++f)
    {
      output[query_index * kHeadDim + f * WARP_SIZE + lane] =
        static_cast<T>(fragment[f] * normalization);
    }
    if (logsumexp != nullptr && lane == 0)
    {
      logsumexp[query_index] = static_cast<T>(
        max_value + Numerics::MathFunctions::get_natural_log<AccT>(sum));
    }
  }
}

//------------------------------------------------------------------------------
/// Host-side launcher: one warp per query row, kWarpsPerBlock rows per
/// block; grid y spans the flattened (batch, head) slices. Pass
/// logsumexp = nullptr unless the backward pass will run.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
void flash_attention_warp_cooperative(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length,
  const int number_of_batch_heads = 1)
{
  constexpr int WARP_SIZE {32};
  const int number_of_row_blocks {
    (sequence_length + kWarpsPerBlock - 1) / kWarpsPerBlock};
  const dim3 grid {
    static_cast<unsigned int>(number_of_row_blocks),
    static_cast<unsigned int>(number_of_batch_heads)};
  flash_attention_forward_warp_cooperative<
    T,
    kHeadDim,
    kWarpsPerBlock,
    kCausal><<<grid, kWarpsPerBlock * WARP_SIZE>>>(
      output,
      logsumexp,
      queries,
      keys,
      values,
      sequence_length);
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_WARP_COOPERATIVE_H
