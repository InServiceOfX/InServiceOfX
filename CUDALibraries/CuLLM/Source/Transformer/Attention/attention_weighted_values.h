#ifndef TRANSFORMER_ATTENTION_ATTENTION_WEIGHTED_VALUES_H
#define TRANSFORMER_ATTENTION_ATTENTION_WEIGHTED_VALUES_H

#include "Transformer/Softmax/AccumulationType.h"

namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
/// Computes the attention output of scaled dot-product attention
/// (see the section on Scaled Dot-Product Attention in FlashAttention.tex):
///
///   O := P V  ∈ R^{n×d_v},
///
/// where P ∈ R^{n×n} is the attention weight matrix (each row on the
/// probability simplex, P_i = softmax(S_i)) and V ∈ R^{n×d_v} holds the value
/// vectors as rows.
///
/// Row i of the output is
///
///   O_i = Σ_{j=1}^{n} P_ij · v_j  ∈ conv{v_1, ..., v_n} ⊂ R^{d_v},
///
/// a convex combination of the value rows — attention as a soft
/// nearest-neighbour lookup: position i retrieves a weighted average of
/// values, weighted by the probability that key k_j best matches query q_i.
///
/// Launch configuration: one block per output row i (gridDim.x = n).
///
/// Memory access: each thread owns output dimensions
/// a = threadIdx.x, threadIdx.x + blockDim.x, ... and folds over all j. At a
/// fixed j, consecutive threads read consecutive V[j·d_v + a] — coalesced —
/// while every thread reads the same weight P[i·n + j] — a warp broadcast.
/// The accumulator lives in a register; O is written once, coalesced.
///
/// kHeadDim = d_v must be a compile-time constant, matching
/// AttentionAccumulator<AccT, kHeadDim>. T is the I/O type;
/// AccT = accumulation_type_t<T> is the accumulation precision.
//------------------------------------------------------------------------------
template <typename T, int kHeadDim>
__global__ void attention_weighted_values(
  T* output,
  const T* weights,
  const T* values,
  const int sequence_length)
{
  using AccT = Softmax::accumulation_type_t<T>;

  const int query_index {static_cast<int>(blockIdx.x)};
  const T* weight_row {weights + query_index * sequence_length};

  // O_ia = Σ_j P_ij · V_ja for a = threadIdx.x, threadIdx.x + blockDim.x, ...
  for (
    int a {static_cast<int>(threadIdx.x)};
    a < kHeadDim;
    a += static_cast<int>(blockDim.x))
  {
    AccT accumulated {0};
    for (int j {0}; j < sequence_length; ++j)
    {
      accumulated += static_cast<AccT>(weight_row[j]) *
        static_cast<AccT>(values[j * kHeadDim + a]);
    }

    output[query_index * kHeadDim + a] = static_cast<T>(accumulated);
  }
}

} // namespace Attention
} // namespace Transformer

#endif // TRANSFORMER_ATTENTION_ATTENTION_WEIGHTED_VALUES_H
