#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/multi_head_attention.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::multi_head_attention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_mha_inputs(const int count, const int seed)
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
// Full double-precision CPU reference implementing the definition of MHA in
// FlashAttention.tex directly from the learned weight matrices:
//   MHA(y) = [head_1 | ... | head_h] W^O,  head_l = Att(yW^Q_l, yW^K_l, yW^V_l).
// Independently derived from the GPU pipeline's stages (fused QKV GEMM,
// per-head safe-softmax attention, concatenation, output linear-map GEMM)
// rather than reusing any of this library's device kernels, so the comparison
// is a genuine end-to-end check of the whole composition.
//------------------------------------------------------------------------------
vector<float> multi_head_attention_cpu(
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const vector<float>& output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int head_dim,
  const int sequence_length,
  const bool causal)
{
  const int d_model {num_heads * head_dim};
  const int num_tokens {batch_size * sequence_length};
  const double scale {1.0 / std::sqrt(static_cast<double>(head_dim))};

  // qkv = input @ qkv_weight_matrix.
  vector<double> qkv(static_cast<size_t>(num_tokens) * 3 * d_model, 0.0);
  for (int row {0}; row < num_tokens; ++row)
  {
    for (int col {0}; col < 3 * d_model; ++col)
    {
      double accumulated {0.0};
      for (int k {0}; k < d_model; ++k)
      {
        accumulated += static_cast<double>(input[row * d_model + k]) *
          static_cast<double>(qkv_weight_matrix[k * 3 * d_model + col]);
      }
      qkv[static_cast<size_t>(row) * 3 * d_model + col] = accumulated;
    }
  }

  // Per-head safe-softmax attention, writing directly into the
  // (num_tokens, d_model) concatenated layout output_weight_matrix expects.
  vector<double> concatenated(static_cast<size_t>(num_tokens) * d_model, 0.0);
  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
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
            const double q_val {
              qkv[static_cast<size_t>(q_row) * 3 * d_model + h * head_dim +
                d]};
            const double k_val {
              qkv[static_cast<size_t>(k_row) * 3 * d_model + d_model +
                h * head_dim + d]};
            dot += q_val * k_val;
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
            const double v_val {
              qkv[static_cast<size_t>(v_row) * 3 * d_model + 2 * d_model +
                h * head_dim + d]};
            accumulated += (weights[j] / sum) * v_val;
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
// End-to-end test harness: allocates every workspace buffer documented in
// multi_head_attention.h, runs the full device pipeline, and returns the
// host-side output.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, bool kCausal>
vector<float> run_multi_head_attention(
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const vector<float>& output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int sequence_length)
{
  const int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};
  const int per_head_elements {batch_size * num_heads * sequence_length * kHD};

  Array<float> d_input(num_tokens * d_model);
  Array<float> d_qkv_weights(d_model * 3 * d_model);
  Array<float> d_output_weights(d_model * d_model);
  Array<float> d_qkv_workspace(num_tokens * 3 * d_model);
  Array<float> d_queries(per_head_elements);
  Array<float> d_keys(per_head_elements);
  Array<float> d_values(per_head_elements);
  Array<float> d_attention_output(per_head_elements);
  Array<float> d_concat_workspace(num_tokens * d_model);
  Array<float> d_output(num_tokens * d_model);

  d_input.copy_host_input_to_device(input);
  d_qkv_weights.copy_host_input_to_device(qkv_weight_matrix);
  d_output_weights.copy_host_input_to_device(output_weight_matrix);

  LibraryContextHandle handle {};
  Stream stream {};

  const bool success {multi_head_attention<float, kHD, kWarps, kCausal>(
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
    sequence_length)};
  EXPECT_TRUE(success);
  cudaDeviceSynchronize();

  vector<float> output(num_tokens * d_model);
  d_output.copy_device_output_to_host(output);
  return output;
}

//------------------------------------------------------------------------------
// Full pipeline (fused QKV linear-map GEMM -> per-head flash attention ->
// merge -> output linear-map GEMM) against the from-scratch CPU reference.
// Sizes chosen for multiple flash-attention tiles and multiple warps per block.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionTests, MatchesCpuReference)
{
  constexpr int kHD {32};
  constexpr int kWarps {4};
  constexpr int batch_size {2};
  constexpr int num_heads {3};
  constexpr int sequence_length {48};
  constexpr int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};

  const vector<float> input {make_mha_inputs(num_tokens * d_model, 3)};
  const vector<float> qkv_weight_matrix {
    make_mha_inputs(d_model * 3 * d_model, 5)};
  const vector<float> output_weight_matrix {
    make_mha_inputs(d_model * d_model, 11)};

  const vector<float> output {run_multi_head_attention<kHD, kWarps, false>(
    input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads,
    sequence_length)};

  const vector<float> expected {multi_head_attention_cpu(
    input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads, kHD,
    sequence_length, false)};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i {0}; i < output.size(); ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 5e-3f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Causal variant: masking flows through qkv_linear_maps unchanged (it knows
// nothing about masking) into flash_attention_warp_cooperative's kCausal.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionTests, CausalMatchesCpuReference)
{
  constexpr int kHD {32};
  constexpr int kWarps {4};
  constexpr int batch_size {2};
  constexpr int num_heads {3};
  constexpr int sequence_length {48};
  constexpr int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};

  const vector<float> input {make_mha_inputs(num_tokens * d_model, 13)};
  const vector<float> qkv_weight_matrix {
    make_mha_inputs(d_model * 3 * d_model, 17)};
  const vector<float> output_weight_matrix {
    make_mha_inputs(d_model * d_model, 19)};

  const vector<float> output {run_multi_head_attention<kHD, kWarps, true>(
    input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads,
    sequence_length)};

  const vector<float> expected {multi_head_attention_cpu(
    input, qkv_weight_matrix, output_weight_matrix, batch_size, num_heads, kHD,
    sequence_length, true)};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i {0}; i < output.size(); ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 5e-3f) << "index " << i;
  }
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
