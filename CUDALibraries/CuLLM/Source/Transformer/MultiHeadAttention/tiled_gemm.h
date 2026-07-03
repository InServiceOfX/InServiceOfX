#ifndef TRANSFORMER_MULTI_HEAD_ATTENTION_TILED_GEMM_H
#define TRANSFORMER_MULTI_HEAD_ATTENTION_TILED_GEMM_H

namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
/// Hand-written tiled shared-memory GEMM: the pedagogical correctness
/// baseline for the cuBLASLt-backed linear maps (qkv_linear_maps.h,
/// output_linear_map.h). Computes the same row-major right-multiplication
///
///   Out(m,n) = X(m,k) · W(k,n)
///
/// as one kernel, with the canonical square-tile decomposition: each thread
/// block owns a kTile×kTile tile of Out; the k dimension is walked in
/// kTile-wide steps, each step staging one X tile and one W tile through
/// shared memory so every global element is read once per tile pass instead
/// of once per output element — a k/kTile-fold reduction in global traffic
/// over the naive kernel.
///
/// Per k-step, thread (ty, tx) of the block:
///   1. loads X[block_row·kTile + ty][step·kTile + tx] and
///            W[step·kTile + ty][block_col·kTile + tx]   (both coalesced:
///      tx walks the fastest-varying dimension of both row-major operands),
///   2. synchronizes,
///   3. accumulates Σ_j x_tile[ty][j] · w_tile[j][tx] from shared memory,
///   4. synchronizes before the next step overwrites the tiles.
///
/// This is deliberately the textbook version — no register blocking, no
/// double buffering, no tensor cores — so its measured gap against cuBLASLt
/// (see Benchmarks/Transformer/MultiHeadAttention/linear_map_gemm_benchmark)
/// quantifies what the library implementation buys. Out-of-range guards
/// zero-fill the shared tiles, so m, n, k need not be tile multiples.
//------------------------------------------------------------------------------
template <typename T, int kTile>
__global__ void tiled_gemm(
  T* output,
  const T* input,
  const T* weight_matrix,
  const int m,
  const int k,
  const int n)
{
  __shared__ T input_tile[kTile][kTile];
  __shared__ T weight_tile[kTile][kTile];

  const int thread_row {static_cast<int>(threadIdx.y)};
  const int thread_column {static_cast<int>(threadIdx.x)};
  const int row {static_cast<int>(blockIdx.y) * kTile + thread_row};
  const int column {static_cast<int>(blockIdx.x) * kTile + thread_column};

  T accumulated {0};

  const int number_of_steps {(k + kTile - 1) / kTile};
  for (int step {0}; step < number_of_steps; ++step)
  {
    const int input_column {step * kTile + thread_column};
    input_tile[thread_row][thread_column] =
      (row < m && input_column < k) ?
        input[row * k + input_column] : T{0};

    const int weight_row {step * kTile + thread_row};
    weight_tile[thread_row][thread_column] =
      (weight_row < k && column < n) ?
        weight_matrix[weight_row * n + column] : T{0};

    __syncthreads();

    #pragma unroll
    for (int j {0}; j < kTile; ++j)
    {
      accumulated += input_tile[thread_row][j] * weight_tile[j][thread_column];
    }

    __syncthreads();
  }

  if (row < m && column < n)
  {
    output[row * n + column] = accumulated;
  }
}

//------------------------------------------------------------------------------
/// Host-side launcher: one thread per output element, kTile×kTile threads
/// per block.
//------------------------------------------------------------------------------
template <typename T, int kTile = 32>
void tiled_gemm_launch(
  T* output,
  const T* input,
  const T* weight_matrix,
  const int m,
  const int k,
  const int n)
{
  const dim3 block {kTile, kTile};
  const dim3 grid {
    static_cast<unsigned int>((n + kTile - 1) / kTile),
    static_cast<unsigned int>((m + kTile - 1) / kTile)};
  tiled_gemm<T, kTile><<<grid, block>>>(
    output, input, weight_matrix, m, k, n);
}

} // namespace MultiHeadAttention
} // namespace Transformer

#endif // TRANSFORMER_MULTI_HEAD_ATTENTION_TILED_GEMM_H
