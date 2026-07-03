#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/multi_head_attention.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
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
// End-to-end __nv_bfloat16 instantiation of the full MHA pipeline. bfloat16
// is 1 sign + 8 exponent + 7 mantissa bits — float's exponent range at
// ~2^-8 relative precision, requiring sm_80+ for native math and BF16
// tensor cores.
//
// The GEMMs run CUDA_R_16BF data under CUBLAS_COMPUTE_32F with CUDA_R_32F
// (float) alpha/beta — there is no CUBLAS_COMPUTE_16BF — exercising the
// scale-type split between ComputeParameters::data_type_ and scale_type_
// and LtMatrixMultiplication's host_scale_type_t. The attention core
// touches bfloat16 only at the I/O casts; all arithmetic runs in float via
// AccumulationType<__nv_bfloat16> (primary template → float).
//
// Inputs are quantized to bfloat16 BEFORE the CPU reference runs, so only
// the pipeline's arithmetic error is measured. Because the GEMMs accumulate
// in FP32 (unlike __half's COMPUTE_16F), the dominant error is bfloat16
// rounding at each stage boundary (~2^-8 relative), not accumulation drift.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-0.5, 0.5], pre-rounded to
// bfloat16.
//------------------------------------------------------------------------------
vector<float> make_bf16_quantized_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    const float value {
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 210.0f};
    result[i] = __bfloat162float(__float2bfloat16(value));
  }
  return result;
}

//------------------------------------------------------------------------------
// Double-precision CPU reference (same math as
// multi_head_attention_tests.cu's; duplicated so this file stands alone in
// its own translation unit).
//------------------------------------------------------------------------------
vector<float> multi_head_attention_cpu_for_bf16(
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const vector<float>& output_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int head_dim,
  const int sequence_length)
{
  const int d_model {num_heads * head_dim};
  const int num_tokens {batch_size * sequence_length};
  const double scale {1.0 / std::sqrt(static_cast<double>(head_dim))};

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

  vector<double> concatenated(static_cast<size_t>(num_tokens) * d_model, 0.0);
  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
      for (int i {0}; i < sequence_length; ++i)
      {
        vector<double> scores(sequence_length);
        for (int j {0}; j < sequence_length; ++j)
        {
          const int q_row {b * sequence_length + i};
          const int k_row {b * sequence_length + j};
          double dot {0.0};
          for (int d {0}; d < head_dim; ++d)
          {
            dot +=
              qkv[static_cast<size_t>(q_row) * 3 * d_model + h * head_dim + d]
              * qkv[static_cast<size_t>(k_row) * 3 * d_model + d_model +
                  h * head_dim + d];
          }
          scores[j] = dot * scale;
        }
        const double max_score {
          *std::max_element(scores.begin(), scores.end())};
        double sum {0.0};
        vector<double> weights(sequence_length);
        for (int j {0}; j < sequence_length; ++j)
        {
          weights[j] = std::exp(scores[j] - max_score);
          sum += weights[j];
        }

        const size_t out_row {static_cast<size_t>(b) * sequence_length + i};
        for (int d {0}; d < head_dim; ++d)
        {
          double accumulated {0.0};
          for (int j {0}; j < sequence_length; ++j)
          {
            const int v_row {b * sequence_length + j};
            accumulated += (weights[j] / sum) *
              qkv[static_cast<size_t>(v_row) * 3 * d_model + 2 * d_model +
                h * head_dim + d];
          }
          concatenated[out_row * d_model + h * head_dim + d] = accumulated;
        }
      }
    }
  }

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
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionBfloat16Tests, Bfloat16PipelineMatchesCpuReference)
{
  constexpr int kHD {32};
  constexpr int kWarps {4};
  constexpr int batch_size {2};
  constexpr int num_heads {2};
  constexpr int sequence_length {32};
  constexpr int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};
  const int per_head_elements {batch_size * num_heads * sequence_length * kHD};

  const vector<float> input {
    make_bf16_quantized_inputs(num_tokens * d_model, 3)};
  const vector<float> qkv_weights {
    make_bf16_quantized_inputs(d_model * 3 * d_model, 5)};
  const vector<float> output_weights {
    make_bf16_quantized_inputs(d_model * d_model, 11)};

  const auto to_bf16 {[](const vector<float>& values)
  {
    vector<__nv_bfloat16> result(values.size());
    for (size_t i {0}; i < values.size(); ++i)
    {
      result[i] = __float2bfloat16(values[i]);
    }
    return result;
  }};

  Array<__nv_bfloat16> d_input(num_tokens * d_model);
  Array<__nv_bfloat16> d_qkv_weights(d_model * 3 * d_model);
  Array<__nv_bfloat16> d_output_weights(d_model * d_model);
  Array<__nv_bfloat16> d_qkv_workspace(num_tokens * 3 * d_model);
  Array<__nv_bfloat16> d_queries(per_head_elements);
  Array<__nv_bfloat16> d_keys(per_head_elements);
  Array<__nv_bfloat16> d_values(per_head_elements);
  Array<__nv_bfloat16> d_attention_output(per_head_elements);
  Array<__nv_bfloat16> d_concat_workspace(num_tokens * d_model);
  Array<__nv_bfloat16> d_output(num_tokens * d_model);
  Array<__nv_bfloat16> d_logsumexp(batch_size * num_heads * sequence_length);

  {
    vector<__nv_bfloat16> h {to_bf16(input)};
    d_input.copy_host_input_to_device(h);
  }
  {
    vector<__nv_bfloat16> h {to_bf16(qkv_weights)};
    d_qkv_weights.copy_host_input_to_device(h);
  }
  {
    vector<__nv_bfloat16> h {to_bf16(output_weights)};
    d_output_weights.copy_host_input_to_device(h);
  }

  LibraryContextHandle handle {};
  Stream stream {};

  const bool success {multi_head_attention<__nv_bfloat16, kHD, kWarps, false>(
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
    sequence_length,
    d_logsumexp.elements_)};
  ASSERT_TRUE(success);
  cudaDeviceSynchronize();

  vector<__nv_bfloat16> output_bf16(num_tokens * d_model);
  d_output.copy_device_output_to_host(output_bf16);

  const vector<float> expected {multi_head_attention_cpu_for_bf16(
    input, qkv_weights, output_weights, batch_size, num_heads, kHD,
    sequence_length)};

  // bfloat16 rounds each stage boundary at ~2^-8 relative; the FP32 GEMM
  // accumulation contributes far less. Outputs have magnitude up to a few
  // units, so the bound is dominated by the relative term.
  double max_error {0.0};
  for (size_t i {0}; i < output_bf16.size(); ++i)
  {
    const double got {
      static_cast<double>(__bfloat162float(output_bf16[i]))};
    const double difference {std::abs(got - expected[i])};
    max_error = std::max(max_error, difference);
    ASSERT_NEAR(got, expected[i], 1e-1 + 5e-2 * std::abs(expected[i]))
      << "index " << i;
  }
  // The observed error must sit in a genuinely bfloat16 band — a float-level
  // max error would mean the 16BF path silently fell back to float.
  EXPECT_GT(max_error, 1e-6);
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
