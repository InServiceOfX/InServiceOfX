#ifndef TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_H
#define TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_H

#include "Numerics/Constants/get_infinity.h"
#include "Numerics/MathFunctions.h"
#include "Transformer/Attention/AttentionAccumulator.h"
#include "Transformer/Softmax/AccumulationType.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// FlashAttention forward pass: O = Att(Q,K,V) = softmax(QK^⊤/√d_k)V in a
/// single fused kernel that never writes the n×n matrices S or P to HBM
/// (see the section on The FlashAttention Algorithm in FlashAttention.tex).
///
/// The key insight: the attention output accumulator α(A) = (m, ℓ, õ) is a
/// commutative-monoid element (AttentionAccumulator), so output row i is the
/// normalized result of the left fold over K/V column tiles
///
///   α({1,...,n}) = ⊕_{j=1}^{T_c} α(A_j),   A_j = {(j−1)B_c + 1, ..., j·B_c},
///
/// and each α(A_j) depends only on q_i, K_j, V_j — all resident in SRAM
/// (shared memory). Q, K, V, O are each read/written from HBM Θ(1) times per
/// tile pass, giving Θ(n²d/M) HBM traffic instead of standard attention's
/// Θ(n²) (see the sections on IO Complexity of Standard Attention and of
/// FlashAttention in FlashAttention.tex).
///
/// Tile parameters (compile-time): kBlockRows = B_r query rows per thread
/// block, kBlockColumns = B_c key/value rows per shared-memory tile. The SRAM
/// constraint B_r·d_k + B_c·(d_k + d_v) ≤ M becomes: the Q, K, V tiles below
/// must fit in shared memory (e.g. d = 64, B_r = 64, B_c = 32 → 33 KB).
///
/// Thread mapping: one thread per query row (blockDim.x must equal
/// kBlockRows; gridDim.x = ⌈n/B_r⌉). Each thread owns the full accumulator
/// (m_i, ℓ_i, õ_i ∈ R^{d_v}) for its row in registers — the same sequential
/// use AttentionAccumulator was designed for. Per K/V tile, the thread
/// computes its row of S_ij = Q_i K_j^⊤/√d_k and folds the tile's
/// contribution into the running accumulator.
///
/// The tile update below is the row-wise merge of the running accumulator
/// with the tile accumulator α(A_j) = (m_ij, ℓ_ij, õ_ij)
/// (see the section on The Attention Output Accumulator in
/// FlashAttention.tex), fused so õ_ij never needs its own registers:
///
///   m_new = max(m, m_ij)
///   ℓ_new = e^{m − m_new}·ℓ + e^{m_ij − m_new}·ℓ_ij
///   õ_new = e^{m − m_new}·õ + e^{m_ij − m_new}·õ_ij
///
/// Because e^{s − m_ij}·e^{m_ij − m_new} = e^{s − m_new}, the tile's terms
/// can be exponentiated against m_new directly and FMA'd straight into
/// (ℓ, õ): rescale the running accumulator once by e^{m − m_new}, then for
/// each key j in the tile add p_j = e^{s_j − m_new} to ℓ and p_j·V_j to õ.
/// One rescale per tile instead of one per key — algebraically identical to
/// iterated merge() (which the unit tests verify against).
///
/// Identity handling: m is initialized to −∞ (attention_identity), and on
/// the first tile e^{−∞ − m_new} = 0 rescales the empty accumulator away —
/// no NaN, since m_new is finite once the tile holds ≥ 1 key. Keys past the
/// end of the sequence in a partial last tile are padded with score −∞, i.e.
/// p_j = 0: the monoid identity contributes nothing.
///
/// Epilogue: O_i = õ/ℓ (the normalized output o(A) of the accumulator),
/// written to HBM once — the only n×d_v write of the whole kernel.
///
/// Causal (autoregressive) masking, kCausal = true: the causal mask
/// M_ij = 0 for i ≥ j, −∞ for i < j restricts row i's attention weights to
/// a distribution supported on {1, ..., i} (see the definition of the
/// causal mask in the section on The Decoder Stack in FlashAttention.tex).
/// In tiled form the mask acts at two granularities:
///   - Tile level: a K/V tile whose first key index exceeds the block's
///     last query row is masked for *every* row of the Q block, so the
///     inner loop simply stops early — about half the tiles (and half the
///     K/V HBM reads) are skipped. The bound depends only on blockIdx, so
///     all threads of the block exit the loop together (barrier-safe).
///   - Element level: within a straddling diagonal tile, keys with
///     tile_start + j > query_index get score −∞ — identical to the
///     partial-tile padding, p_j = 0, the monoid identity. Key 0 is never
///     masked (0 ≤ i for every row), so every row's fold still meets at
///     least one finite score in the first tile.
///
/// Batch and multi-head support: in multi-head attention each head applies
/// Att independently to its own (Q_h, K_h, V_h) slice (see the section on
/// Multi-Head Attention in FlashAttention.tex) — the heads only interact in
/// the input projections and the output concatenation, which are GEMMs
/// outside this kernel. Independence maps directly onto the grid:
/// blockIdx.y indexes the flattened (batch, head) pair, and Q, K, V, O are
/// laid out as (batch·heads, n, d) with each slice contiguous — llm.c's
/// "permuted" layout, chosen so a slice's rows stay coalesced. gridDim.y = 1
/// recovers single-head attention unchanged.
///
/// kHeadDim = d_k = d_v; T is the I/O type;
/// AccT = accumulation_type_t<T> is the accumulation precision.
//------------------------------------------------------------------------------
template <
  typename T,
  int kHeadDim,
  int kBlockRows,
  int kBlockColumns,
  bool kCausal = false>
__global__ void flash_attention_forward(
  T* output,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;

  static_assert(kBlockRows > 0 && kBlockColumns > 0 && kHeadDim > 0);

  // Each (batch, head) slice is an independent attention problem; offset
  // every pointer to this block's slice.
  const int slice_offset {
    static_cast<int>(blockIdx.y) * sequence_length * kHeadDim};
  output += slice_offset;
  queries += slice_offset;
  keys += slice_offset;
  values += slice_offset;

  // Q tile is padded by one column: thread t reads row t repeatedly, and for
  // kHeadDim a multiple of the warp size an unpadded stride would put every
  // thread's row in the same shared-memory bank (32-way conflicts).
  // K and V tiles need no padding: at a fixed (j, d) every thread reads the
  // same element — a broadcast.
  __shared__ AccT shared_queries[kBlockRows][kHeadDim + 1];
  __shared__ AccT shared_keys[kBlockColumns][kHeadDim];
  __shared__ AccT shared_values[kBlockColumns][kHeadDim];

  const int row_in_block {static_cast<int>(threadIdx.x)};
  const int query_index {
    static_cast<int>(blockIdx.x) * kBlockRows + row_in_block};

  // Load Q_i from HBM to SRAM once (coalesced cooperative load); it is
  // reused by every tile of the inner loop. Rows past the end of the
  // sequence are zero-filled; their threads never write output.
  for (
    int index {static_cast<int>(threadIdx.x)};
    index < kBlockRows * kHeadDim;
    index += static_cast<int>(blockDim.x))
  {
    const int row {index / kHeadDim};
    const int d {index % kHeadDim};
    const int global_row {
      static_cast<int>(blockIdx.x) * kBlockRows + row};
    shared_queries[row][d] = (global_row < sequence_length) ?
      static_cast<AccT>(queries[global_row * kHeadDim + d]) : AccT{0};
  }

  const AccT scale {
    AccT{1} /
      Numerics::MathFunctions::get_sqrt<AccT>(static_cast<AccT>(kHeadDim))};

  const AccT negative_infinity {
    -Numerics::Constants::get_infinity<AccT>()};

  // Running accumulator (m_i, ℓ_i, õ_i), initialized to the monoid identity
  // (−∞, 0, 0_{d_v}). õ_i lives in registers because kHeadDim is a
  // compile-time constant.
  AttentionAccumulator<AccT, kHeadDim> accumulator {
    attention_identity<AccT, kHeadDim>()};

  int number_of_tiles {
    (sequence_length + kBlockColumns - 1) / kBlockColumns};
  if (kCausal)
  {
    // Tiles whose first key index exceeds the block's last query row are
    // fully masked for every row of this Q block — skip them. Uniform
    // across the block (depends only on blockIdx), so the __syncthreads()
    // calls below stay aligned.
    const int last_query_in_block {
      static_cast<int>(blockIdx.x) * kBlockRows + kBlockRows - 1};
    const int last_needed_tile {last_query_in_block / kBlockColumns};
    number_of_tiles = (last_needed_tile + 1 < number_of_tiles) ?
      last_needed_tile + 1 : number_of_tiles;
  }

  for (int tile {0}; tile < number_of_tiles; ++tile)
  {
    const int tile_start {tile * kBlockColumns};
    const int tile_size {
      (sequence_length - tile_start < kBlockColumns) ?
        sequence_length - tile_start : kBlockColumns};

    // All threads must be done reading the previous K/V tiles (and, on the
    // first iteration, done writing the Q tile) before overwriting.
    __syncthreads();

    // Load K_j, V_j from HBM to SRAM (coalesced cooperative load); rows past
    // the end of the sequence are zero-filled so the −∞-padded score path
    // below multiplies 0·0, never 0·garbage.
    for (
      int index {static_cast<int>(threadIdx.x)};
      index < kBlockColumns * kHeadDim;
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

    // Row of S_ij: s_j = q_i · k_j / √d_k for each key j in the tile, and
    // the row max m_ij. Compile-time bounds keep scores[] in registers;
    // padded keys get score −∞ (the empty-element convention).
    AccT scores[kBlockColumns];
    AccT tile_max {negative_infinity};
    #pragma unroll
    for (int j {0}; j < kBlockColumns; ++j)
    {
      // Out-of-sequence keys and causally masked keys (M_ij = −∞ for
      // j > i) both take the −∞ score path: p_j = 0, the monoid identity.
      const bool masked {kCausal && (tile_start + j > query_index)};
      if (j < tile_size && !masked)
      {
        AccT dot {0};
        #pragma unroll
        for (int d {0}; d < kHeadDim; ++d)
        {
          dot += shared_queries[row_in_block][d] * shared_keys[j][d];
        }
        scores[j] = dot * scale;
        tile_max = Numerics::MathFunctions::get_max<AccT>(
          tile_max,
          scores[j]);
      }
      else
      {
        scores[j] = negative_infinity;
      }
    }

    // Row-wise merge with the tile accumulator, fused (see header comment):
    // rescale the running (ℓ, õ) once by e^{m − m_new}, then accumulate the
    // tile's p_j = e^{s_j − m_new} terms directly.
    const AccT new_max {Numerics::MathFunctions::get_max<AccT>(
      accumulator.max_value,
      tile_max)};
    // e^{−∞ − m_new} = 0 on the first tile: the identity rescales away.
    const AccT rescale {Numerics::MathFunctions::get_exponential<AccT>(
      accumulator.max_value - new_max)};

    accumulator.max_value = new_max;
    accumulator.sum *= rescale;
    #pragma unroll
    for (int d {0}; d < kHeadDim; ++d)
    {
      accumulator.output[d] *= rescale;
    }

    #pragma unroll
    for (int j {0}; j < kBlockColumns; ++j)
    {
      // p_j = e^{s_j − m_new}; 0 for −∞-padded keys.
      const AccT p {Numerics::MathFunctions::get_exponential<AccT>(
        scores[j] - new_max)};
      accumulator.sum += p;
      #pragma unroll
      for (int d {0}; d < kHeadDim; ++d)
      {
        accumulator.output[d] += p * shared_values[j][d];
      }
    }
  }

  // Epilogue: O_i = õ/ℓ, the normalized output of the folded accumulator —
  // the kernel's only output write to HBM.
  if (query_index < sequence_length)
  {
    const AccT normalization {AccT{1} / accumulator.sum};
    #pragma unroll
    for (int d {0}; d < kHeadDim; ++d)
    {
      output[query_index * kHeadDim + d] = static_cast<T>(
        accumulator.output[d] * normalization);
    }
  }
}

//------------------------------------------------------------------------------
/// Host-side launcher: one thread per query row, one block per B_r rows;
/// grid y spans the flattened (batch, head) slices, e.g.
/// number_of_batch_heads = B·NH for batch size B and NH heads, with Q, K, V,
/// O laid out as (B·NH, n, d).
//------------------------------------------------------------------------------
template <
  typename T,
  int kHeadDim,
  int kBlockRows,
  int kBlockColumns,
  bool kCausal = false>
void flash_attention(
  T* output,
  const T* queries,
  const T* keys,
  const T* values,
  const int sequence_length,
  const int number_of_batch_heads = 1)
{
  const int number_of_row_blocks {
    (sequence_length + kBlockRows - 1) / kBlockRows};
  const dim3 grid {
    static_cast<unsigned int>(number_of_row_blocks),
    static_cast<unsigned int>(number_of_batch_heads)};
  flash_attention_forward<T, kHeadDim, kBlockRows, kBlockColumns, kCausal>
    <<<grid, kBlockRows>>>(
      output,
      queries,
      keys,
      values,
      sequence_length);
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_FLASH_ATTENTION_FORWARD_H
