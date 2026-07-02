#include "DataStructures/Array.h"
#include "Transformer/Attention/attention_weighted_values.h"
#include "gtest/gtest.h"

#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::attention_weighted_values;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// P = identity: row i puts all weight on value row i, so O = V. The extreme
// (vertex-of-the-simplex) case of the convex combination
// O_i = Σ_j P_ij v_j ∈ conv{v_1, ..., v_n}.
//------------------------------------------------------------------------------
TEST(AttentionWeightedValuesTests, IdentityWeightsReturnValues)
{
  constexpr int n {4};
  constexpr int kHD {8};

  vector<float> weights(n * n, 0.0f);
  for (int i {0}; i < n; ++i)
  {
    weights[i * n + i] = 1.0f;
  }

  vector<float> values(n * kHD);
  for (int i {0}; i < n * kHD; ++i)
  {
    values[i] = static_cast<float>(i);
  }

  Array<float> d_weights(n * n);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  d_weights.copy_host_input_to_device(weights);
  d_values.copy_host_input_to_device(values);

  attention_weighted_values<float, kHD><<<n, 128>>>(
    d_output.elements_, d_weights.elements_, d_values.elements_, n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < n * kHD; ++i)
  {
    EXPECT_NEAR(output[i], values[i], 1e-6f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Uniform weights P_ij = 1/n: every output row is the barycenter (mean) of
// the value rows — the center of the simplex, the other extreme from one-hot.
//------------------------------------------------------------------------------
TEST(AttentionWeightedValuesTests, UniformWeightsGiveMeanOfValueRows)
{
  constexpr int n {4};
  constexpr int kHD {4};

  const vector<float> weights(n * n, 1.0f / static_cast<float>(n));

  // Value rows: v_j = (j+1) · [1, 1, 1, 1]; mean over j is
  // (1+2+3+4)/4 = 2.5 in every component.
  vector<float> values(n * kHD);
  for (int j {0}; j < n; ++j)
  {
    for (int a {0}; a < kHD; ++a)
    {
      values[j * kHD + a] = static_cast<float>(j + 1);
    }
  }

  Array<float> d_weights(n * n);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  d_weights.copy_host_input_to_device(weights);
  d_values.copy_host_input_to_device(values);

  attention_weighted_values<float, kHD><<<n, 128>>>(
    d_output.elements_, d_weights.elements_, d_values.elements_, n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < n * kHD; ++i)
  {
    EXPECT_NEAR(output[i], 2.5f, 1e-6f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// kHD larger than blockDim exercises the strided output-dimension loop
// (each thread owns multiple output dimensions). Compare to a double-
// precision CPU fold of O_ia = Σ_j P_ij V_ja.
//------------------------------------------------------------------------------
TEST(AttentionWeightedValuesTests, MatchesCpuReferenceWithStridedDimensions)
{
  constexpr int n {16};
  constexpr int kHD {96};
  constexpr int block_size {32};

  vector<float> weights(n * n);
  for (int i {0}; i < n; ++i)
  {
    // Unnormalized positive weights, then normalize the row so it lies on
    // the probability simplex like a genuine softmax output.
    float row_sum {0.0f};
    for (int j {0}; j < n; ++j)
    {
      weights[i * n + j] = static_cast<float>((i * 5 + j * 3) % 7 + 1);
      row_sum += weights[i * n + j];
    }
    for (int j {0}; j < n; ++j)
    {
      weights[i * n + j] /= row_sum;
    }
  }

  vector<float> values(n * kHD);
  for (int i {0}; i < n * kHD; ++i)
  {
    values[i] = static_cast<float>((i * 11 + 2) % 19 - 9) / 9.0f;
  }

  Array<float> d_weights(n * n);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  d_weights.copy_host_input_to_device(weights);
  d_values.copy_host_input_to_device(values);

  attention_weighted_values<float, kHD><<<n, block_size>>>(
    d_output.elements_, d_weights.elements_, d_values.elements_, n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < n; ++i)
  {
    for (int a {0}; a < kHD; ++a)
    {
      double expected {0.0};
      for (int j {0}; j < n; ++j)
      {
        expected += static_cast<double>(weights[i * n + j]) *
          static_cast<double>(values[j * kHD + a]);
      }
      ASSERT_NEAR(
        output[i * kHD + a], static_cast<float>(expected), 1e-5f)
        << "entry (" << i << ", " << a << ")";
    }
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
