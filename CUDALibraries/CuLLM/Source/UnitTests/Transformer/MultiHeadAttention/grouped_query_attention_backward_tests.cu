#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/Attention/flash_attention_backward.h"
#include "Transformer/MultiHeadAttention/grouped_query_attention.h"
#include "Transformer/MultiHeadAttention/grouped_query_attention_backward.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::Attention::reduce_grouped_kv_gradients;
using Transformer::MultiHeadAttention::grouped_query_attention;
using Transformer::MultiHeadAttention::grouped_query_attention_backward;
using Transformer::MultiHeadAttention::merge_grouped_qkv_heads;
using Transformer::MultiHeadAttention::split_grouped_qkv_heads;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-0.5, 0.5].
//------------------------------------------------------------------------------
vector<float> make_gqa_backward_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 210.0f;
  }
  return result;
}

//------------------------------------------------------------------------------
// The group sum computed exactly on the CPU: with g partial slices per KV
// head, every output element must equal the sum of its g inputs.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionBackwardTests, ReductionSumsEachGroup)
{
  constexpr int kHD {32};
  constexpr int batch_size {2};
  constexpr int num_heads {4};
  constexpr int kv_group_size {2};
  constexpr int num_kv_heads {num_heads / kv_group_size};
  constexpr int sequence_length {5};
  const int partial_elements {
    batch_size * num_heads * sequence_length * kHD};
  const int reduced_elements {
    batch_size * num_kv_heads * sequence_length * kHD};

  const vector<float> partial_keys {
    make_gqa_backward_inputs(partial_elements, 3)};
  const vector<float> partial_values {
    make_gqa_backward_inputs(partial_elements, 7)};

  Array<float> d_partial_keys(partial_elements);
  Array<float> d_partial_values(partial_elements);
  Array<float> d_keys(reduced_elements);
  Array<float> d_values(reduced_elements);
  d_partial_keys.copy_host_input_to_device(partial_keys);
  d_partial_values.copy_host_input_to_device(partial_values);

  constexpr int kThreadsPerBlock {256};
  const int blocks {
    (reduced_elements + kThreadsPerBlock - 1) / kThreadsPerBlock};
  reduce_grouped_kv_gradients<float, kHD><<<blocks, kThreadsPerBlock>>>(
    d_keys.elements_,
    d_values.elements_,
    d_partial_keys.elements_,
    d_partial_values.elements_,
    batch_size,
    num_heads,
    kv_group_size,
    sequence_length);
  cudaDeviceSynchronize();

  vector<float> keys(reduced_elements), values(reduced_elements);
  d_keys.copy_device_output_to_host(keys);
  d_values.copy_device_output_to_host(values);

  const int slice {sequence_length * kHD};
  for (int b {0}; b < batch_size; ++b)
  {
    for (int kv {0}; kv < num_kv_heads; ++kv)
    {
      for (int e {0}; e < slice; ++e)
      {
        float key_sum {0.0f};
        float value_sum {0.0f};
        for (int j {0}; j < kv_group_size; ++j)
        {
          const int partial_index {
            ((b * num_heads + kv * kv_group_size + j) * slice) + e};
          key_sum += partial_keys[partial_index];
          value_sum += partial_values[partial_index];
        }
        const int out_index {(b * num_kv_heads + kv) * slice + e};
        ASSERT_NEAR(keys[out_index], key_sum, 1e-6f)
          << "b=" << b << " kv=" << kv << " e=" << e;
        ASSERT_NEAR(values[out_index], value_sum, 1e-6f)
          << "b=" << b << " kv=" << kv << " e=" << e;
      }
    }
  }
}

//------------------------------------------------------------------------------
// merge_grouped_qkv_heads must be the exact inverse of
// split_grouped_qkv_heads: scattering (dQ, dK, dV) into the fused layout
// and splitting again must reproduce all three tensors bit for bit.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionBackwardTests,
  MergeGroupedInvertsSplitGrouped)
{
  constexpr int kHD {32};
  constexpr int batch_size {2};
  constexpr int num_heads {4};
  constexpr int kv_group_size {2};
  constexpr int num_kv_heads {num_heads / kv_group_size};
  constexpr int sequence_length {5};
  const int fused_width {(num_heads + 2 * num_kv_heads) * kHD};
  const int fused_elements {batch_size * sequence_length * fused_width};
  const int query_elements {batch_size * num_heads * sequence_length * kHD};
  const int kv_elements {batch_size * num_kv_heads * sequence_length * kHD};

  const vector<float> queries_in {make_gqa_backward_inputs(query_elements, 3)};
  const vector<float> keys_in {make_gqa_backward_inputs(kv_elements, 5)};
  const vector<float> values_in {make_gqa_backward_inputs(kv_elements, 11)};

  Array<float> d_queries(query_elements);
  Array<float> d_keys(kv_elements);
  Array<float> d_values(kv_elements);
  Array<float> d_fused(fused_elements);
  Array<float> d_queries_out(query_elements);
  Array<float> d_keys_out(kv_elements);
  Array<float> d_values_out(kv_elements);
  d_queries.copy_host_input_to_device(queries_in);
  d_keys.copy_host_input_to_device(keys_in);
  d_values.copy_host_input_to_device(values_in);

  constexpr int kThreadsPerBlock {256};
  const int blocks {
    (query_elements + kThreadsPerBlock - 1) / kThreadsPerBlock};

  merge_grouped_qkv_heads<float, kHD><<<blocks, kThreadsPerBlock>>>(
    d_fused.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    batch_size,
    num_heads,
    kv_group_size,
    sequence_length);
  split_grouped_qkv_heads<float, kHD><<<blocks, kThreadsPerBlock>>>(
    d_queries_out.elements_,
    d_keys_out.elements_,
    d_values_out.elements_,
    d_fused.elements_,
    batch_size,
    num_heads,
    kv_group_size,
    sequence_length);
  cudaDeviceSynchronize();

  vector<float> queries_out(query_elements);
  vector<float> keys_out(kv_elements);
  vector<float> values_out(kv_elements);
  d_queries_out.copy_device_output_to_host(queries_out);
  d_keys_out.copy_device_output_to_host(keys_out);
  d_values_out.copy_device_output_to_host(values_out);

  for (int i {0}; i < query_elements; ++i)
  {
    ASSERT_EQ(queries_out[i], queries_in[i]) << "query index " << i;
  }
  for (int i {0}; i < kv_elements; ++i)
  {
    ASSERT_EQ(keys_out[i], keys_in[i]) << "key index " << i;
    ASSERT_EQ(values_out[i], values_in[i]) << "value index " << i;
  }
}

//------------------------------------------------------------------------------
// Gradient-check fixture, following multi_head_attention_backward_tests:
// loss L := ⟨G, Y⟩ = Σ G ⊙ GQA(X), so dL/dY = G is the gradient_output and
// every analytic parameter gradient is checked against the central
// difference of the *device* forward — exercising the whole grouped chain
// including the per-query-head partials and the group-sum reduction.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, int kGroupSize, bool kCausal>
class GqaGradientCheck
{
  public:

    static constexpr int batch_size {1};
    static constexpr int num_heads {4};
    static constexpr int num_kv_heads {num_heads / kGroupSize};
    static constexpr int sequence_length {8};
    static constexpr int d_model {num_heads * kHD};
    static constexpr int fused_width {
      (num_heads + 2 * num_kv_heads) * kHD};
    static constexpr int num_tokens {batch_size * sequence_length};
    static constexpr int query_elements {
      batch_size * num_heads * sequence_length * kHD};
    static constexpr int kv_elements {
      batch_size * num_kv_heads * sequence_length * kHD};

    GqaGradientCheck():
      input_{make_gqa_backward_inputs(num_tokens * d_model, 3)},
      qkv_weights_{make_gqa_backward_inputs(d_model * fused_width, 5)},
      output_weights_{make_gqa_backward_inputs(d_model * d_model, 11)},
      loss_weights_{make_gqa_backward_inputs(num_tokens * d_model, 13)}
    {}

    double loss(
      const vector<float>& input,
      const vector<float>& qkv_weights,
      const vector<float>& output_weights)
    {
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
      d_qkv_weights.copy_host_input_to_device(qkv_weights);
      d_output_weights.copy_host_input_to_device(output_weights);

      LibraryContextHandle handle {};
      Stream stream {};
      EXPECT_TRUE((grouped_query_attention<float, kHD, kWarps, kCausal>(
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
        kGroupSize,
        sequence_length)));
      cudaDeviceSynchronize();

      vector<float> output(num_tokens * d_model);
      d_output.copy_device_output_to_host(output);

      double total {0.0};
      for (size_t i {0}; i < output.size(); ++i)
      {
        total += static_cast<double>(loss_weights_[i]) *
          static_cast<double>(output[i]);
      }
      return total;
    }

    void analytic_gradients(
      vector<float>& gradient_input,
      vector<float>& gradient_qkv_weights,
      vector<float>& gradient_output_weights)
    {
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
      Array<float> d_logsumexp(batch_size * num_heads * sequence_length);

      d_input.copy_host_input_to_device(input_);
      d_qkv_weights.copy_host_input_to_device(qkv_weights_);
      d_output_weights.copy_host_input_to_device(output_weights_);

      LibraryContextHandle handle {};
      Stream stream {};
      ASSERT_TRUE((grouped_query_attention<float, kHD, kWarps, kCausal>(
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
        kGroupSize,
        sequence_length,
        d_logsumexp.elements_)));
      cudaDeviceSynchronize();

      Array<float> d_gradient_output(num_tokens * d_model);
      Array<float> d_gradient_input(num_tokens * d_model);
      Array<float> d_gradient_qkv_weights(d_model * fused_width);
      Array<float> d_gradient_output_weights(d_model * d_model);
      Array<float> d_gradient_concat(num_tokens * d_model);
      Array<float> d_gradient_attention_output(query_elements);
      Array<float> d_gradient_queries(query_elements);
      Array<float> d_partial_gradient_keys(query_elements);
      Array<float> d_partial_gradient_values(query_elements);
      Array<float> d_gradient_keys(kv_elements);
      Array<float> d_gradient_values(kv_elements);
      Array<float> d_gradient_qkv(num_tokens * fused_width);
      Array<float> d_row_dots(batch_size * num_heads * sequence_length);

      d_gradient_output.copy_host_input_to_device(loss_weights_);

      ASSERT_TRUE((
        grouped_query_attention_backward<float, kHD, kWarps, kCausal>(
          handle,
          stream,
          d_gradient_input.elements_,
          d_gradient_qkv_weights.elements_,
          d_gradient_output_weights.elements_,
          d_concat_workspace.elements_,
          d_gradient_concat.elements_,
          d_gradient_attention_output.elements_,
          d_gradient_queries.elements_,
          d_partial_gradient_keys.elements_,
          d_partial_gradient_values.elements_,
          d_gradient_keys.elements_,
          d_gradient_values.elements_,
          d_gradient_qkv.elements_,
          d_row_dots.elements_,
          d_gradient_output.elements_,
          d_input.elements_,
          d_qkv_weights.elements_,
          d_output_weights.elements_,
          d_queries.elements_,
          d_keys.elements_,
          d_values.elements_,
          d_attention_output.elements_,
          d_logsumexp.elements_,
          batch_size,
          num_heads,
          kGroupSize,
          sequence_length)));
      cudaDeviceSynchronize();

      gradient_input.resize(num_tokens * d_model);
      gradient_qkv_weights.resize(d_model * fused_width);
      gradient_output_weights.resize(d_model * d_model);
      d_gradient_input.copy_device_output_to_host(gradient_input);
      d_gradient_qkv_weights.copy_device_output_to_host(gradient_qkv_weights);
      d_gradient_output_weights.copy_device_output_to_host(
        gradient_output_weights);
    }

    enum class Tensor { kInput, kQkvWeights, kOutputWeights };

    double finite_difference(const Tensor which, const size_t index)
    {
      const float eps {1e-2f};
      vector<float> input {input_};
      vector<float> qkv_weights {qkv_weights_};
      vector<float> output_weights {output_weights_};
      vector<float>& target {
        which == Tensor::kInput ? input :
        which == Tensor::kQkvWeights ? qkv_weights : output_weights};

      const float saved {target[index]};
      target[index] = saved + eps;
      const double loss_plus {loss(input, qkv_weights, output_weights)};
      target[index] = saved - eps;
      const double loss_minus {loss(input, qkv_weights, output_weights)};
      return (loss_plus - loss_minus) / (2.0 * static_cast<double>(eps));
    }

    void check_all(const int probes_per_tensor)
    {
      vector<float> gradient_input;
      vector<float> gradient_qkv_weights;
      vector<float> gradient_output_weights;
      analytic_gradients(
        gradient_input, gradient_qkv_weights, gradient_output_weights);

      struct Target
      {
        Tensor which;
        const vector<float>* analytic;
        const char* name;
      };
      const Target targets[3] {
        {Tensor::kInput, &gradient_input, "input"},
        {Tensor::kQkvWeights, &gradient_qkv_weights, "qkv_weights"},
        {Tensor::kOutputWeights, &gradient_output_weights, "output_weights"}};

      for (const Target& target : targets)
      {
        const size_t size {target.analytic->size()};
        const size_t stride {size / probes_per_tensor + 1};
        for (size_t index {0}; index < size; index += stride)
        {
          const double numeric {finite_difference(target.which, index)};
          const double analytic {
            static_cast<double>((*target.analytic)[index])};
          const double tolerance {2e-2 + 5e-2 * std::abs(numeric)};
          EXPECT_NEAR(analytic, numeric, tolerance)
            << target.name << " index " << index;
        }
      }
    }

    vector<float> input_;
    vector<float> qkv_weights_;
    vector<float> output_weights_;
    vector<float> loss_weights_;
};

//------------------------------------------------------------------------------
// Grouped-query (4 query heads, 2 shared K/V pairs): every analytic
// gradient — including dW^K/dW^V columns, which exist only via the group
// sum — must match central differences of the device forward.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionBackwardTests, GradientsMatchFiniteDifferences)
{
  GqaGradientCheck<32, 4, 2, false> check {};
  check.check_all(12);
}

//------------------------------------------------------------------------------
// Causal grouped-query: masking composes with the group K/V sharing in the
// recomputed weights.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionBackwardTests,
  CausalGradientsMatchFiniteDifferences)
{
  GqaGradientCheck<32, 4, 2, true> check {};
  check.check_all(12);
}

//------------------------------------------------------------------------------
// Multi-query (g = NH = 4, one shared pair): the reduction sums all four
// query heads' partials into the single dK/dV.
//------------------------------------------------------------------------------
TEST(GroupedQueryAttentionBackwardTests,
  MultiQueryGradientsMatchFiniteDifferences)
{
  GqaGradientCheck<32, 4, 4, false> check {};
  check.check_all(12);
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
