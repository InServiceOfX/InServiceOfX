#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/output_projection.h"
#include "gtest/gtest.h"

#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::output_projection;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_output_projection_inputs(const int count, const int seed)
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
// CPU reference: merge_heads (concatenate columns per token), then
// Out = Concat @ W^O (row-major, double-precision accumulation) — the
// MHA(y) := [head_1|...|head_h] W^O output projection.
//------------------------------------------------------------------------------
vector<float> output_projection_cpu(
  const vector<float>& per_head_output,
  const vector<float>& output_weights,
  const int batch_size,
  const int num_heads,
  const int head_dim,
  const int sequence_length)
{
  const int d_model {num_heads * head_dim};
  const int num_tokens {batch_size * sequence_length};

  vector<double> concatenated(static_cast<size_t>(num_tokens) * d_model);
  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
      for (int t {0}; t < sequence_length; ++t)
      {
        const size_t in_index {
          static_cast<size_t>((b * num_heads + h) * sequence_length + t) *
            head_dim};
        const size_t out_row {static_cast<size_t>(b) * sequence_length + t};
        for (int d {0}; d < head_dim; ++d)
        {
          concatenated[out_row * d_model + h * head_dim + d] =
            static_cast<double>(per_head_output[in_index + d]);
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
          static_cast<double>(output_weights[k * d_model + col]);
      }
      output[static_cast<size_t>(row) * d_model + col] =
        static_cast<float>(accumulated);
    }
  }
  return output;
}

//------------------------------------------------------------------------------
// Full merge + GEMM against the CPU reference.
//------------------------------------------------------------------------------
TEST(OutputProjectionTests, MatchesCpuReference)
{
  constexpr int kHD {16};
  constexpr int batch_size {2};
  constexpr int num_heads {4};
  constexpr int sequence_length {20};
  constexpr int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};

  const vector<float> per_head_output {
    make_output_projection_inputs(
      batch_size * num_heads * sequence_length * kHD, 7)};
  const vector<float> output_weights {
    make_output_projection_inputs(d_model * d_model, 11)};

  Array<float> d_per_head_output(
    batch_size * num_heads * sequence_length * kHD);
  Array<float> d_weights(d_model * d_model);
  Array<float> d_workspace(num_tokens * d_model);
  Array<float> d_output(num_tokens * d_model);
  d_per_head_output.copy_host_input_to_device(per_head_output);
  d_weights.copy_host_input_to_device(output_weights);

  LibraryContextHandle handle {};
  Stream stream {};

  ASSERT_TRUE((output_projection<float, kHD>(
    handle,
    stream,
    d_output.elements_,
    d_workspace.elements_,
    d_per_head_output.elements_,
    d_weights.elements_,
    batch_size,
    num_heads,
    sequence_length)));
  cudaDeviceSynchronize();

  vector<float> output(d_output.number_of_elements_);
  d_output.copy_device_output_to_host(output);

  const vector<float> expected {output_projection_cpu(
    per_head_output, output_weights, batch_size, num_heads, kHD,
    sequence_length)};

  ASSERT_EQ(output.size(), expected.size());
  for (size_t i {0}; i < output.size(); ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 1e-3f) << "index " << i;
  }
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
