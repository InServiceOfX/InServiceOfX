#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_CUTE_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_CUTE_H

#if defined(CULLM_HAS_CUTLASS)

#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// FlashAttention forward pass built on CUTLASS CuTe — the next rung of the
/// optimization ladder above flash_attention_tensor_core.h (WMMA). Same
/// FA-2 math throughout; three engine upgrades, each targeting a cost the
/// WMMA kernel measurably pays:
///
///   1. Register-resident output accumulator. CuTe's coordinate tensors
///      (make_identity_tensor + partition_C) tell every lane which (row,
///      column) of the accumulator each of its fragment registers holds —
///      the information WMMA's opaque fragments withhold. The per-row
///      rescale by exp(m_old − m_new) is applied directly to the live
///      fragment, and P·V accumulates into it via mma. The WMMA kernel's
///      per-tile store→merge→(reload) round trip of the 16×d õ through
///      shared memory disappears.
///   2. cp.async double-buffered K/V tiles (SM80_CP_ASYNC_CACHEGLOBAL,
///      128-bit per thread): the next tile's global→shared copy overlaps
///      the current tile's math instead of serializing before it.
///   3. Q fragments are loaded from shared to registers once and reused by
///      every K/V tile (the WMMA kernel re-issues load_matrix_sync per
///      tile).
///
/// Retained from the WMMA design: one warp per 16-row query tile,
/// SM80 16×8×16 half MMA atoms with float accumulators (tiled to
/// 16×16×16), online softmax on the score tile in shared memory (one lane
/// per row — O(16²) scalar work against the matmuls' O(16²·d)), P rounded
/// to half before P·V, causal tile skipping with longest-block-first
/// scheduling, and the logsumexp epilogue.
///
/// T must be __half; kHeadDim must equal 64 in this version — the cp.async
/// tiled copy shape (128 threads × 8 halves = one 16×64 tile per issue) and
/// the benchmark shapes are both built around d = 64. kWarpsPerBlock must
/// be 4 so blockDim.x = 128 matches that copy shape. Requires sm_80+ at
/// runtime (cp.async and the SM80 mma atom); compiles under sm_75 but traps
/// if launched there.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
__global__ void flash_attention_forward_cute(
  T* output,
  T* logsumexp,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length)
{
  static_assert(
    std::is_same_v<T, __half>,
    "flash_attention_forward_cute currently supports __half only");
  static_assert(
    kHeadDim == 64,
    "the cp.async tiled-copy shape is built for kHeadDim == 64");
  static_assert(
    kWarpsPerBlock == 4,
    "the cp.async tiled-copy shape needs blockDim.x == 128");

  using namespace cute;

  constexpr int WARP_SIZE {32};
  constexpr int kTile {16};
  constexpr int kBlockRows {kWarpsPerBlock * kTile};
  constexpr int kStages {2};

  const int warp_rank {static_cast<int>(threadIdx.x) / WARP_SIZE};
  const int lane {static_cast<int>(threadIdx.x) % WARP_SIZE};

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

  // Longest causal row blocks first, as in the other attention kernels.
  const int row_block {kCausal ?
    static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x) :
    static_cast<int>(blockIdx.x)};
  const int block_row_start {row_block * kBlockRows};
  const int warp_row_start {block_row_start + warp_rank * kTile};

  // Shared memory: double-buffered K/V tiles (block-wide), per-warp Q
  // staging, score tile, half P tile, and per-row softmax statistics.
  __shared__ __half shared_keys[kStages][kTile][kHeadDim];
  __shared__ __half shared_values[kStages][kTile][kHeadDim];
  __shared__ __half shared_queries[kWarpsPerBlock][kTile][kHeadDim];
  __shared__ float shared_scores[kWarpsPerBlock][kTile][kTile];
  __shared__ __half shared_weights[kWarpsPerBlock][kTile][kTile];
  __shared__ float shared_max[kWarpsPerBlock][kTile];
  __shared__ float shared_sum[kWarpsPerBlock][kTile];
  __shared__ float shared_rescale[kWarpsPerBlock][kTile];

  const float negative_infinity {
    -Numerics::Constants::get_infinity<float>()};
  const float scale {
    1.0f / Numerics::MathFunctions::get_sqrt<float>(
      static_cast<float>(kHeadDim))};

  // -- Q staging: global -> shared (scalar, once), zero-padded ------------
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
  if (lane < kTile)
  {
    shared_max[warp_rank][lane] = negative_infinity;
    shared_sum[warp_rank][lane] = 0.0f;
  }
  __syncthreads();

  // -- CuTe MMA setup ------------------------------------------------------
  // One warp; SM80 16x8x16 half->float atom replayed to a 16x16x16 tile.
  TiledMMA tiled_mma {make_tiled_mma(
    SM80_16x8x16_F32F16F16F32_TN{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<kTile>, Int<kTile>, Int<kTile>>{})};
  ThrMMA thr_mma {tiled_mma.get_thread_slice(lane)};

  // Q as the A operand of S = Q K^T: (M, K) = (16, kHeadDim), K-major (the
  // "T" of TN). Partitioned into registers once, reused for every tile.
  Tensor q_smem {make_tensor(
    make_smem_ptr(reinterpret_cast<half_t*>(&shared_queries[warp_rank][0][0])),
    Layout<Shape<Int<kTile>, Int<kHeadDim>>,
           Stride<Int<kHeadDim>, _1>>{})};
  Tensor q_fragment {thr_mma.partition_fragment_A(q_smem)};
  copy(thr_mma.partition_A(q_smem), q_fragment);

  // Running output accumulator õ: a live C fragment over (16, kHeadDim),
  // with a coordinate tensor telling each lane its elements' (row, column).
  Tensor output_shape {make_identity_tensor(
    Shape<Int<kTile>, Int<kHeadDim>>{})};
  Tensor output_coordinates {thr_mma.partition_C(output_shape)};
  Tensor output_fragment {thr_mma.partition_fragment_C(
    make_tensor(
      make_smem_ptr(static_cast<float*>(nullptr)),
      Layout<Shape<Int<kTile>, Int<kHeadDim>>>{}))};
  clear(output_fragment);

  // Score tile S as a C fragment over (16, 16), plus its coordinates.
  Tensor score_shape {make_identity_tensor(Shape<Int<kTile>, Int<kTile>>{})};
  Tensor score_coordinates {thr_mma.partition_C(score_shape)};

  // -- cp.async tiled copy for K/V tiles -----------------------------------
  // 128 threads x 8 halves (16 bytes) = one 16x64 tile per issue.
  TiledCopy tiled_copy {make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>{},
    Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    Layout<Shape<_1, _8>>{})};
  ThrCopy thr_copy {tiled_copy.get_thread_slice(static_cast<int>(threadIdx.x))};

  const auto issue_tile_copy {[&](const int tile, const int stage)
  {
    const int tile_start {tile * kTile};
    if (tile_start + kTile <= sequence_length)
    {
      // Full tile: 128-bit cp.async, overlapping with compute.
      Tensor k_global {make_tensor(
        make_gmem_ptr(reinterpret_cast<const half_t*>(
          keys + tile_start * kHeadDim)),
        Layout<Shape<Int<kTile>, Int<kHeadDim>>,
               Stride<Int<kHeadDim>, _1>>{})};
      Tensor v_global {make_tensor(
        make_gmem_ptr(reinterpret_cast<const half_t*>(
          values + tile_start * kHeadDim)),
        Layout<Shape<Int<kTile>, Int<kHeadDim>>,
               Stride<Int<kHeadDim>, _1>>{})};
      Tensor k_smem {make_tensor(
        make_smem_ptr(reinterpret_cast<half_t*>(&shared_keys[stage][0][0])),
        Layout<Shape<Int<kTile>, Int<kHeadDim>>,
               Stride<Int<kHeadDim>, _1>>{})};
      Tensor v_smem {make_tensor(
        make_smem_ptr(reinterpret_cast<half_t*>(&shared_values[stage][0][0])),
        Layout<Shape<Int<kTile>, Int<kHeadDim>>,
               Stride<Int<kHeadDim>, _1>>{})};
      copy(tiled_copy, thr_copy.partition_S(k_global),
        thr_copy.partition_D(k_smem));
      copy(tiled_copy, thr_copy.partition_S(v_global),
        thr_copy.partition_D(v_smem));
    }
    else
    {
      // Ragged tail tile: scalar guarded loads (once per kernel at most).
      for (
        int index {static_cast<int>(threadIdx.x)};
        index < kTile * kHeadDim;
        index += static_cast<int>(blockDim.x))
      {
        const int row {index / kHeadDim};
        const int d {index % kHeadDim};
        const int global_row {tile_start + row};
        const bool in_range {global_row < sequence_length};
        shared_keys[stage][row][d] = in_range ?
          keys[global_row * kHeadDim + d] : __half{0};
        shared_values[stage][row][d] = in_range ?
          values[global_row * kHeadDim + d] : __half{0};
      }
    }
    cp_async_fence();
  }};

  int number_of_tiles {(sequence_length + kTile - 1) / kTile};
  if (kCausal)
  {
    const int last_query_in_block {block_row_start + kBlockRows - 1};
    const int last_needed_tile {last_query_in_block / kTile};
    number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
      last_needed_tile + 1 : number_of_tiles;
  }

  // Prologue: start the first tile's copy before entering the loop.
  issue_tile_copy(0, 0);

  for (int tile {0}; tile < number_of_tiles; ++tile)
  {
    const int stage {tile % kStages};
    const int tile_start {tile * kTile};

    if (tile + 1 < number_of_tiles)
    {
      // Overlap: issue tile+1 into the other stage, then wait for tile's
      // copy only (<1> = at most one copy group still in flight). The
      // __syncthreads() at the end of the previous iteration guaranteed
      // every warp finished consuming the stage being overwritten.
      issue_tile_copy(tile + 1, (tile + 1) % kStages);
      cp_async_wait<1>();
    }
    else
    {
      cp_async_wait<0>();
    }
    __syncthreads();

    // -- S tile = Q · K_tile^T on tensor cores -----------------------------
    // K stored row-major (key, d) is exactly the (N, K)-shaped K-major B
    // operand of the TN atom.
    Tensor k_smem {make_tensor(
      make_smem_ptr(reinterpret_cast<half_t*>(&shared_keys[stage][0][0])),
      Layout<Shape<Int<kTile>, Int<kHeadDim>>,
             Stride<Int<kHeadDim>, _1>>{})};
    Tensor k_fragment {thr_mma.partition_fragment_B(k_smem)};
    copy(thr_mma.partition_B(k_smem), k_fragment);

    Tensor score_fragment {thr_mma.partition_fragment_C(
      make_tensor(
        make_smem_ptr(static_cast<float*>(nullptr)),
        Layout<Shape<Int<kTile>, Int<kTile>>>{}))};
    clear(score_fragment);
    gemm(tiled_mma, q_fragment, k_fragment, score_fragment);

    // Fragment -> shared, coordinates making the scatter explicit.
    CUTE_UNROLL
    for (int i {0}; i < size(score_fragment); ++i)
    {
      shared_scores[warp_rank]
        [get<0>(score_coordinates(i))][get<1>(score_coordinates(i))] =
          score_fragment(i);
    }
    __syncwarp();

    // -- Online softmax, one lane per query row ----------------------------
    if (lane < kTile)
    {
      const int query_index {warp_row_start + lane};
      float tile_max {negative_infinity};
      CUTE_UNROLL
      for (int j {0}; j < kTile; ++j)
      {
        const int key_index {tile_start + j};
        const bool masked {
          key_index >= sequence_length ||
          (kCausal && key_index > query_index)};
        if (!masked)
        {
          tile_max = Numerics::MathFunctions::get_max<float>(
            tile_max, shared_scores[warp_rank][lane][j] * scale);
        }
      }

      const float old_max {shared_max[warp_rank][lane]};
      const float new_max {
        Numerics::MathFunctions::get_max<float>(old_max, tile_max)};
      const float rescale {(old_max == negative_infinity) ?
        0.0f :
        Numerics::MathFunctions::get_approximate_exponential<float>(
          old_max - new_max)};

      float row_sum {0.0f};
      CUTE_UNROLL
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
              shared_scores[warp_rank][lane][j] * scale - new_max);
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

    // -- Rescale the LIVE output fragment, then accumulate P·V into it ----
    // This is what the coordinate tensor buys over WMMA: each lane knows
    // the row of every register it holds, so the per-row factor applies
    // in place, and mma accumulates on top — no shared round trip for õ.
    CUTE_UNROLL
    for (int i {0}; i < size(output_fragment); ++i)
    {
      output_fragment(i) *=
        shared_rescale[warp_rank][get<0>(output_coordinates(i))];
    }

    Tensor p_smem {make_tensor(
      make_smem_ptr(reinterpret_cast<half_t*>(&shared_weights[warp_rank][0][0])),
      Layout<Shape<Int<kTile>, Int<kTile>>, Stride<Int<kTile>, _1>>{})};
    Tensor p_fragment {thr_mma.partition_fragment_A(p_smem)};
    copy(thr_mma.partition_A(p_smem), p_fragment);

    // V as the B operand of õ += P V: (N, K) = (kHeadDim, keys). V is
    // stored row-major (key, d), so this view is its transpose — same
    // bytes-are-the-transpose relabelling as everywhere else in CuLLM.
    Tensor v_smem {make_tensor(
      make_smem_ptr(reinterpret_cast<half_t*>(&shared_values[stage][0][0])),
      Layout<Shape<Int<kHeadDim>, Int<kTile>>,
             Stride<_1, Int<kHeadDim>>>{})};
    Tensor v_fragment {thr_mma.partition_fragment_B(v_smem)};
    copy(thr_mma.partition_B(v_smem), v_fragment);

    gemm(tiled_mma, p_fragment, v_fragment, output_fragment);

    // All warps done with this stage before the copy issued next iteration
    // overwrites it.
    __syncthreads();
  }

  // -- Epilogue: O = õ/ℓ straight from registers, L = m + ln ℓ -------------
  CUTE_UNROLL
  for (int i {0}; i < size(output_fragment); ++i)
  {
    const int row {static_cast<int>(get<0>(output_coordinates(i)))};
    const int d {static_cast<int>(get<1>(output_coordinates(i)))};
    const int query_index {warp_row_start + row};
    if (query_index < sequence_length)
    {
      const float sum {shared_sum[warp_rank][row]};
      output[query_index * kHeadDim + d] = __float2half(
        (sum > 0.0f) ? output_fragment(i) / sum : 0.0f);
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
/// Host-side launcher, mirroring flash_attention_tensor_core.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim, int kWarpsPerBlock, bool kCausal = false>
void flash_attention_cute(
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
  flash_attention_forward_cute<T, kHeadDim, kWarpsPerBlock, kCausal>
    <<<grid, kWarpsPerBlock * WARP_SIZE>>>(
      output,
      logsumexp,
      queries,
      keys,
      values,
      sequence_length);
}

} // namespace Attention
} // namespace Transformer

#endif // CULLM_HAS_CUTLASS
#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_CUTE_H
