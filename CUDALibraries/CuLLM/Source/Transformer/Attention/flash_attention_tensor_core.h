#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_TENSOR_CORE_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_TENSOR_CORE_H

#include <cuda_fp16.h>
#include <mma.h>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// FlashAttention forward pass with tensor-core (WMMA) tile matmuls.
///
/// Same FA-2 math as flash_attention_warp_cooperative — online softmax over
/// K/V tiles with the (m, ℓ, õ) accumulator, delayed normalization, causal
/// tile skipping, longest-block-first causal scheduling — but the two inner
/// products change engines:
///
///   S_ij = Q_i K_j^⊤   and   õ += P_ij V_j
///
/// are computed as 16×16×16 nvcuda::wmma mma_sync fragment operations
/// (half inputs, float accumulators) instead of one scalar FMA per lane.
/// This targets the gap measured in Documents/AttentionBenchmarkReport.md:
/// the scalar kernel reaches ~6% of FP32 peak while every faster
/// implementation (XLA/cuBLAS, cuDNN) runs its inner products on matmul
/// hardware.
///
/// Execution mapping: one warp per 16-row query tile (WMMA's native M),
/// kWarpsPerBlock warps per block, gridDim.y spanning (batch, head) slices
/// as in every other attention kernel here. Per 16-key K/V tile, a warp:
///
///   1. computes the 16×16 score tile with kHeadDim/16 mma_sync steps and
///      stores it to shared memory;
///   2. runs the online-softmax update on that tile with one lane per query
///      row (scalar — the softmax is O(16²) work against the matmuls'
///      O(16²·d), so leaving it on CUDA cores costs little);
///   3. writes P = exp(S·scale − m_new) to shared as half — the same
///      precision choice cuDNN and FlashAttention-2 make — and computes
///      P·V with kHeadDim/16 more mma_sync steps;
///   4. rescales and merges the tile product into the running õ (scalar,
///      O(16·d)). The rescale-by-exp(m_old − m_new) cannot be applied to a
///      live accumulator fragment because WMMA fragments are opaque (a lane
///      does not know which matrix rows its registers hold), so õ lives in
///      shared memory and each tile's P·V product passes through a
///      store_matrix_sync.
///
/// Numerics: S and õ accumulate in float; only P is rounded to half before
/// the second matmul, so expect fp16-grade output error (~1e-3 relative)
/// against a float reference — the same as cuDNN's flash attention.
///
/// T must be __half (bf16 needs sm_80-only fragment types — future work).
/// kHeadDim must be a multiple of 16 (the WMMA K/N tile), kWarpsPerBlock
/// warps per block cover kWarpsPerBlock·16 query rows. Requires sm_70+.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
__global__ void flash_attention_forward_tensor_core(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length)
{
  static_assert(
    std::is_same_v<T, __half>,
    "flash_attention_forward_tensor_core currently supports __half only");
  static_assert(
    kHeadDim % 16 == 0,
    "WMMA tiling requires kHeadDim to be a multiple of 16");
  static_assert(kWarpsPerBlock > 0);

  namespace wmma = nvcuda::wmma;
  constexpr int WARP_SIZE {32};
  constexpr int kTile {16};
  constexpr int kFragments {kHeadDim / kTile};
  // Query rows covered by one block.
  constexpr int kBlockRows {kWarpsPerBlock * kTile};

  const int warp_rank {static_cast<int>(threadIdx.x) / WARP_SIZE};
  const int lane {static_cast<int>(threadIdx.x) % WARP_SIZE};

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

  // Causal work rebalancing, as in the warp-cooperative kernel: schedule
  // the longest row blocks first.
  const int row_block {kCausal ?
    static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x) :
    static_cast<int>(blockIdx.x)};
  const int block_row_start {row_block * kBlockRows};
  const int warp_row_start {block_row_start + warp_rank * kTile};

  // Shared memory. K/V tiles are shared by all warps of the block; the
  // rest is per-warp. Leading dimensions (kHeadDim and kTile halves/floats)
  // satisfy WMMA's 16-byte multiple requirement.
  __shared__ __half shared_keys[kTile][kHeadDim];
  __shared__ __half shared_values[kTile][kHeadDim];
  __shared__ __half shared_queries[kWarpsPerBlock][kTile][kHeadDim];
  __shared__ __half shared_weights[kWarpsPerBlock][kTile][kTile];
  // The 16×16 score tile and the 16×kHeadDim P·V product tile share one
  // buffer: scores are fully consumed into P (shared_weights) before the
  // product's store_matrix_sync overwrites the region. Saves
  // kWarpsPerBlock·1 KB, which is what fits this kernel under sm_75's
  // 48 KB static shared memory limit at kWarpsPerBlock = 4.
  __shared__ float shared_tile_product[kWarpsPerBlock][kTile][kHeadDim];
  __shared__ float shared_output[kWarpsPerBlock][kTile][kHeadDim];
  __shared__ float shared_max[kWarpsPerBlock][kTile];
  __shared__ float shared_sum[kWarpsPerBlock][kTile];
  __shared__ float shared_rescale[kWarpsPerBlock][kTile];

  const float negative_infinity {
    -Numerics::Constants::get_infinity<float>()};
  const float scale {
    1.0f / Numerics::MathFunctions::get_sqrt<float>(
      static_cast<float>(kHeadDim))};

  // Load this block's Q rows once (cooperative, coalesced); zero-fill
  // out-of-range rows so their fragment math is harmless.
  for (
    int index {static_cast<int>(threadIdx.x)};
    index < kBlockRows * kHeadDim;
    index += static_cast<int>(blockDim.x))
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int global_row {block_row_start + row};
    shared_queries[row / kTile][row % kTile][d] =
      (global_row < sequence_length) ?
        queries[global_row * kHeadDim + d] : __half{0};
  }

  // Initialize the distributed accumulator (m, ℓ, õ) to the identity.
  for (
    int index {static_cast<int>(lane)};
    index < kTile * kHeadDim;
    index += WARP_SIZE)
  {
    shared_output[warp_rank][index / kHeadDim][index % kHeadDim] = 0.0f;
  }
  if (lane < kTile)
  {
    shared_max[warp_rank][lane] = negative_infinity;
    shared_sum[warp_rank][lane] = 0.0f;
  }

  int number_of_tiles {(sequence_length + kTile - 1) / kTile};
  if (kCausal)
  {
    // Skip K/V tiles wholly past the block's last query row (uniform per
    // block, so the __syncthreads() below stay aligned).
    const int last_query_in_block {block_row_start + kBlockRows - 1};
    const int last_needed_tile {last_query_in_block / kTile};
    number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
      last_needed_tile + 1 : number_of_tiles;
  }

  for (int tile {0}; tile < number_of_tiles; ++tile)
  {
    const int tile_start {tile * kTile};

    __syncthreads();
    for (
      int index {static_cast<int>(threadIdx.x)};
      index < kTile * kHeadDim;
      index += static_cast<int>(blockDim.x))
    {
      const int row {index / kHeadDim};
      const int d {index % kHeadDim};
      const int global_row {tile_start + row};
      const bool in_range {global_row < sequence_length};
      shared_keys[row][d] = in_range ?
        keys[global_row * kHeadDim + d] : __half{0};
      shared_values[row][d] = in_range ?
        values[global_row * kHeadDim + d] : __half{0};
    }
    __syncthreads();

    // 1. S tile = Q_warp · K_tile^⊤ on tensor cores. K is stored row-major
    // (key, d); read as a column-major (d, key) fragment it IS K^⊤ — the
    // same bytes-are-the-transpose relabelling as the cuBLASLt linear maps.
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
      score_fragment;
    wmma::fill_fragment(score_fragment, 0.0f);
    #pragma unroll
    for (int f {0}; f < kFragments; ++f)
    {
      wmma::fragment<
        wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major>
        query_fragment;
      wmma::fragment<
        wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major>
        key_fragment;
      wmma::load_matrix_sync(
        query_fragment, &shared_queries[warp_rank][0][f * kTile], kHeadDim);
      wmma::load_matrix_sync(
        key_fragment, &shared_keys[0][f * kTile], kHeadDim);
      wmma::mma_sync(
        score_fragment, query_fragment, key_fragment, score_fragment);
    }
    wmma::store_matrix_sync(
      &shared_tile_product[warp_rank][0][0],
      score_fragment,
      kHeadDim,
      wmma::mem_row_major);
    __syncwarp();

    // 2. Online-softmax update, one lane per query row (lanes 16–31 idle:
    // this step is O(16²) against the matmuls' O(16²·d)).
    if (lane < kTile)
    {
      const int query_index {warp_row_start + lane};
      float tile_max {negative_infinity};
      #pragma unroll
      for (int j {0}; j < kTile; ++j)
      {
        const int key_index {tile_start + j};
        const bool masked {
          key_index >= sequence_length ||
          (kCausal && key_index > query_index)};
        if (!masked)
        {
          const float score {shared_tile_product[warp_rank][lane][j] * scale};
          tile_max = Numerics::MathFunctions::get_max<float>(
            tile_max, score);
        }
      }

      const float old_max {shared_max[warp_rank][lane]};
      const float new_max {
        Numerics::MathFunctions::get_max<float>(old_max, tile_max)};
      // exp(-inf - (-inf)) guard: a fully masked history rescales to 0.
      const float rescale {(old_max == negative_infinity) ?
        0.0f :
        Numerics::MathFunctions::get_approximate_exponential<float>(
          old_max - new_max)};

      float row_sum {0.0f};
      #pragma unroll
      for (int j {0}; j < kTile; ++j)
      {
        const int key_index {tile_start + j};
        const bool masked {
          key_index >= sequence_length ||
          (kCausal && key_index > query_index)};
        float weight {0.0f};
        if (!masked && new_max != negative_infinity)
        {
          weight =
            Numerics::MathFunctions::get_approximate_exponential<float>(
              shared_tile_product[warp_rank][lane][j] * scale - new_max);
        }
        shared_weights[warp_rank][lane][j] = __float2half(weight);
        row_sum += weight;
      }

      shared_max[warp_rank][lane] = new_max;
      shared_sum[warp_rank][lane] =
        shared_sum[warp_rank][lane] * rescale + row_sum;
      shared_rescale[warp_rank][lane] = rescale;
    }
    __syncwarp();

    // 3. Tile product P · V_tile on tensor cores.
    #pragma unroll
    for (int f {0}; f < kFragments; ++f)
    {
      wmma::fragment<
        wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major>
        weight_fragment;
      wmma::fragment<
        wmma::matrix_b, kTile, kTile, kTile, __half, wmma::row_major>
        value_fragment;
      wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
        product_fragment;
      wmma::fill_fragment(product_fragment, 0.0f);
      wmma::load_matrix_sync(
        weight_fragment, &shared_weights[warp_rank][0][0], kTile);
      wmma::load_matrix_sync(
        value_fragment, &shared_values[0][f * kTile], kHeadDim);
      wmma::mma_sync(
        product_fragment, weight_fragment, value_fragment,
        product_fragment);
      wmma::store_matrix_sync(
        &shared_tile_product[warp_rank][0][f * kTile],
        product_fragment,
        kHeadDim,
        wmma::mem_row_major);
    }
    __syncwarp();

    // 4. õ = õ · rescale(row) + P·V, scalar merge (O(16·d) per warp).
    for (
      int index {static_cast<int>(lane)};
      index < kTile * kHeadDim;
      index += WARP_SIZE)
    {
      const int row {index / kHeadDim};
      const int d {index % kHeadDim};
      shared_output[warp_rank][row][d] =
        shared_output[warp_rank][row][d] * shared_rescale[warp_rank][row] +
        shared_tile_product[warp_rank][row][d];
    }
    __syncwarp();
  }

  // Epilogue: O = õ/ℓ and optionally L = m + ln ℓ.
  for (
    int index {static_cast<int>(lane)};
    index < kTile * kHeadDim;
    index += WARP_SIZE)
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int query_index {warp_row_start + row};
    if (query_index < sequence_length)
    {
      const float sum {shared_sum[warp_rank][row]};
      const float normalized {(sum > 0.0f) ?
        shared_output[warp_rank][row][d] / sum : 0.0f};
      output[query_index * kHeadDim + d] = __float2half(normalized);
    }
  }
  if (logsumexp != nullptr && lane < kTile)
  {
    const int query_index {warp_row_start + lane};
    if (query_index < sequence_length)
    {
      const float sum {shared_sum[warp_rank][lane]};
      logsumexp[query_index] = __float2half((sum > 0.0f) ?
        shared_max[warp_rank][lane] +
          Numerics::MathFunctions::get_natural_log<float>(sum) :
        negative_infinity);
    }
  }
}

//------------------------------------------------------------------------------
/// Host-side launcher: kWarpsPerBlock warps per block, each owning a
/// 16-row query tile; grid y spans the flattened (batch, head) slices.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
void flash_attention_tensor_core(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length,
  const int number_of_batch_heads = 1)
{
  constexpr int WARP_SIZE {32};
  constexpr int kBlockRows {kWarpsPerBlock * 16};
  const int number_of_row_blocks {
    (sequence_length + kBlockRows - 1) / kBlockRows};
  const dim3 grid {
    static_cast<unsigned int>(number_of_row_blocks),
    static_cast<unsigned int>(number_of_batch_heads)};
  flash_attention_forward_tensor_core<
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

#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_TENSOR_CORE_H
