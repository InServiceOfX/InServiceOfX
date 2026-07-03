#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/grouped_query_attention.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::grouped_query_attention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_gqa_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

//------------------------------------------------------------------------------
// Double-precision CPU reference implementing the grouped-query remark of
// FlashAttention.tex directly: head ℓ attends with W^Q_ℓ but the *shared*
// W^K_{⌈ℓ/g⌉}, W^V_{⌈ℓ/g⌉}. The fused weight matrix is row-major
// (d_model, (NH + 2·NKV)·head_dim) with column layout
// [Q_0 .. Q_{NH-1} | K_0 .. K_{NKV-1} | V_0 .. V_{NKV-1}].
//------------------------------------------------------------------------------
vector<float> grouped_query_attention_cpu(
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const vector<float>& output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int head_dim,
  const int sequence_length,
  const bool causal)
{
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * head_dim};
  const int fused_width {(num_heads + 2 * num_kv_heads) * head_dim};
  const int num_tokens {batch_size * sequence_length};
  const double scale {1.0 / std::sqrt(static_cast<double>(head_dim))};

  // qkv = input @ qkv_weight_matrix.
  vector<double> qkv(static_cast<size_t>(num_tokens) * fused_width, 0.0);
  for (int row {0}; row < num_tokens; ++row)
  {
    for (int col {0}; col < fused_width; ++col)
    {
      double accumulated {0.0};
      for (int k {0}; k < d_model; ++k)
      {
        accumulated += static_cast<double>(input[row * d_model + k]) *
          static_cast<double>(qkv_weight_matrix[k * fused_width + col]);
      }
      qkv[static_cast<size_t>(row) * fused_width + col] = accumulated;
    }
  }

  // Per-head attention with the group's shared K/V columns.
  vector<double> concatenated(static_cast<size_t>(num_tokens) * d_model, 0.0);
  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
      const int kv_head {h / kv_group_size};
      const int k_column {d_model + kv_head * head_dim};
      const int v_column {d_model + (num_kv_heads + kv_head) * head_dim};
      for (int i {0}; i < sequence_length; ++i)
      {
        const int limit {causal ? i + 1 : sequence_length};
        vector<double> scores(limit);
        for (int j {0}; j < limit; ++j)
        {
          const int q_row {b * sequence_length + i};
          const int k_row {b * sequence_length + j};
          double dot {0.0};
          for (int d {0}; d < head_dim; ++d)
          {
            dot +=
              qkv[static_cast<size_t>(q_row) * fused_width + h * head_dim + d]
              * qkv[static_cast<size_t>(k_row) * fused_width + k_column + d];
          }
          scores[j] = dot * scale;
        }
        const double max_score {
          *std::max_element(scores.begin(), scores.end())};
        double sum {0.0};
        vector<double> weights(limit);
        for (int j {0}; j < limit; ++j)
        {
          weights[j] = std::exp(scores[j] - max_score);
          sum += weights[j];
        }

        const size_t out_row {static_cast<size_t>(b) * sequence_length + i};
        for (int d {0}; d < head_dim; ++d)
        {
          double accumulated {0.0};
          for (int j {0}; j < limit; ++j)
          {
            const int v_row {b * sequence_length + j};
            accumulated += (weights[j] / sum) *
              qkv[static_cast<size_t>(v_row) * fused_width + v_column + d];
          }
          concatenated[out_row * d_model + h * head_dim + d] = accumulated;
        }
      }
    }
  }

  // output = concatenated @ output_weight_matrix.
  vector<float> output(static_cast<size_t>(num_tokens) * d_model);
  for (int row {0}; row < num_tokens; ++row)
  {
    for (int col {0}; col < d_model; ++col)
    {
      double accumulated {0.0};
      for (int k {0}; k < d_model; ++k)
      {
        accumulated += concatenated[static_cast<size_t>(row) * d_model + k] *
          static_cast<double>(output_weight_matrix[k * d_model + col]);
      }
      output[static_cast<size_t>(row) * d_model + col] =
        static_cast<float>(accumulated);
    }
  }
  return output;
}

//------------------------------------------------------------------------------
// Device pipeline harness with the g-dependent buffer shapes.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, bool kCausal>
vector<float> run_grouped_query_attention(
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const vector<float>& output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int sequence_length)
{
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHD};
  const int fused_width {(num_heads + 2 * num_kv_heads) * kHD};
  const int num_tokens {batch_size * sequence_length};
  const int query_elements {batch_size * num_heads * sequence_length * kHD};
  const int kv_elements {batch_size * num_kv_heads * sequence_length * kHD};

  Array<float> d_input(num_tokens * d_model);
  Array<float> d_qkv_weights(d_model * fused_width);
  Array<float> d_output_weights(d_model * d_model);
  Array<float> d_qkv_workspace(num_tokens * fused_width);
  Array<float> d_queries(query_elements);
  Array<float> d_keys(kv_elements);
  Array<float> d_values(kv_elements);
  Array<float> d_attention_output(query_elements);
  Array<float> d_concat_workspace(num_tokens * d_model);
  Array<float> d_output(num_tokens * d_model);

  d_input.copy_host_input_to_device(input);
  d_qkv_weights.copy_host_input_to_device(qkv_weight_matrix);
  d_output_weights.copy_host_input_to_device(output_weight_matrix);

  LibraryContextHandle handle {};
  Stream stream {};

  const bool success {
    grouped_query_attention<float, kHD, kWarps, kCausal>(
      handle,
      stream,
      d_output.elements_,
      d_qkv_workspace.elements_,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      d_attention_output.elements_,
      d_concat_workspace.elements_,
      d_input.elements_,
      d_qkv_weights.elements_,
      d_output_weights.elements_,
      batch_size,
      num_heads,
      kv_group_size,
      sequence_length)};
  EXPECT_TRUE(success);
  cudaDeviceSynchronize();

  vector<float> output(num_tokens * d_model);
  d_output.copy_device_output_to_host(output);
  return output;
}

void compare_against_cpu(
  const int batch_size,
  const int num_heads,
  const int kv_group_size,
  const int sequence_length,
  const bool causal)
{
  constexpr int kHD {32};
  constexpr int kWarps {4};
  const int num_kv_heads {num_heads / kv_group_size};
  const int d_model {num_heads * kHD};
  const int fused_width {(num_heads + 2 * num_kv_heads) * kHD};
  const int num_tokens {batch_size * sequence_length};

  const vector<float> input {make_gqa_inputs(num_tokens * d_model, 3)};
  const vector<float> qkv_weight_matrix {
    make_gqa_inputs(d_model * fused_width, 5)};
  const vector<float> output_weight_matrix {
    make_gqa_inputs(d_model * d_model, 11)};

  const vector<float> output {causal ?
    run_grouped_query_attention<kHD, kWarps, true>(
      input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads,
      kv_group_size, sequence_length) :
    run_grouped_query_attention<kHD, kWarps, false>(
      input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads,
      kv_group_size, sequence_length)};

  const vector<float> expected {grouped_query_attention_cpu(
    input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads,
    kv_group_size, kHD, sequence_length, causal)};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i {0}; i < output.size(); ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 5e-3f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// g = 1 must reproduce standard multi-head attention (the remark's
// degenerate case) — same CPU reference as multi_head_attention_tests.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionTests, GroupSizeOneMatchesMultiHeadAttention)
{
  compare_against_cpu(2, 4, 1, 48, false);
}

//------------------------------------------------------------------------------
// Grouped-query: 4 query heads sharing 2 K/V pairs (g = 2).
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionTests, GroupedMatchesCpuReference)
{
  compare_against_cpu(2, 4, 2, 48, false);
}

//------------------------------------------------------------------------------
// Multi-query: all 4 query heads share a single K/V pair (g = NH).
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionTests, MultiQueryMatchesCpuReference)
{
  compare_against_cpu(2, 4, 4, 48, false);
}

//------------------------------------------------------------------------------
// Causal grouped-query: masking composes with the group K/V sharing.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionTests, CausalGroupedMatchesCpuReference)
{
  compare_against_cpu(2, 4, 2, 48, true);
}

//------------------------------------------------------------------------------
// Non-divisible group size must fail fast rather than compute garbage.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionTests, RejectsNonDivisibleGroupSize)
{
  LibraryContextHandle handle {};
  Stream stream {};
  const bool success {
    grouped_query_attention<float, 32, 4, false>(
      handle,
      stream,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      1,
      /* num_heads = */ 4,
      /* kv_group_size = */ 3,
      8)};
  EXPECT_FALSE(success);
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
