#ifndef TRANSFORMER_ATTENTION_ATTENTION_SCORES_H
#define TRANSFORMER_ATTENTION_ATTENTION_SCORES_H

#include "Numerics/MathFunctions.h"
#include "Transformer/Softmax/AccumulationType.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// Computes the score matrix of scaled dot-product attention
/// (see the section on Scaled Dot-Product Attention in FlashAttention.tex):
///
///   S := Q K^⊤ / √d_k  ∈ R^{n×n},
///
/// where Q ∈ R^{n×d_k} holds the query vectors as rows and K ∈ R^{n×d_k}
/// holds the key vectors as rows. Entry S_ij = q_i · k_j / √d_k is the
/// similarity of query i to key j.
///
/// The 1/√d_k factor (see the section on The Scaling Factor in
/// FlashAttention.tex): if the coordinates of q and k are independent with
/// mean 0 and variance 1, then q · k has mean 0 and variance d_k. Dividing by
/// √d_k restores unit variance, preventing scores from growing with d_k and
/// pushing the softmax toward the simplex vertices (near-zero gradient
/// regime).
///
/// Launch configuration: one block per query row i (gridDim.x = n).
///
/// Data reuse (why shared memory): row S_i needs q_i for all n dot products.
/// Loading q_i from HBM once into shared memory (one coalesced cooperative
/// read) replaces n redundant global reads per row. Each thread then walks
/// key rows j = threadIdx.x, threadIdx.x + blockDim.x, ..., so score writes
/// S[i·n + j] are coalesced across the warp. Key-row reads are strided across
/// threads and rely on L2; the classic fix is a tiled GEMM (or cuBLAS /
/// tensor cores, as llm.c uses), deferred here because FlashAttention
/// restructures this loop entirely — it never materializes S in HBM.
///
/// kHeadDim = d_k must be a compile-time constant so the shared-memory
/// staging buffer is statically sized and the dot-product loop fully unrolls.
/// T is the I/O type; AccT = accumulation_type_t<T> is the dot-product
/// accumulation precision.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void attention_scores(
  T* scores,
  const T* queries,
  const T* keys,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;

  const int query_index {static_cast<int>(blockIdx.x)};

  // Stage q_i in shared memory: one coalesced cooperative load, reused by
  // every thread for all its dot products.
  __shared__ AccT shared_query[kHeadDim];
  for (
    int d {static_cast<int>(threadIdx.x)};
    d < kHeadDim;
    d += static_cast<int>(blockDim.x))
  {
    shared_query[d] =
      static_cast<AccT>(queries[query_index * kHeadDim + d]);
  }
  __syncthreads();

  const AccT scale {
    AccT{1} /
      Numerics::MathFunctions::get_sqrt<AccT>(static_cast<AccT>(kHeadDim))};

  // S_ij = q_i · k_j / √d_k for j = threadIdx.x, threadIdx.x + blockDim.x, ...
  for (
    int j {static_cast<int>(threadIdx.x)};
    j < sequence_length;
    j += static_cast<int>(blockDim.x))
  {
    const T* key_row {keys + j * kHeadDim};

    AccT dot {0};
    #pragma unroll
    for (int d {0}; d < kHeadDim; ++d)
    {
      dot += shared_query[d] * static_cast<AccT>(key_row[d]);
    }

    scores[query_index * sequence_length + j] = static_cast<T>(dot * scale);
  }
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_ATTENTION_SCORES_H
