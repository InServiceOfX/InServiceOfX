#include "DataStructures/Array.h"
#include "Transformer/Attention/scaled_dot_product_attention.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::sqrt;
using std::vector;
using Transformer::Attention::scaled_dot_product_attention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// Double-precision CPU reference for Att(Q,K,V) = softmax(QK^⊤/√d_k)V,
// computed with the safe softmax (subtract the row max before exp).
//------------------------------------------------------------------------------
vector<float> attention_cpu(
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const int n,
  const int d)
{
  vector<float> output(n * d);
  const double scale {1.0 / sqrt(static_cast<double>(d))};

  for (int i {0}; i < n; ++i)
  {
    // S_i = q_i K^⊤ / √d_k
    vector<double> scores(n);
    for (int j {0}; j < n; ++j)
    {
      double dot {0.0};
      for (int c {0}; c < d; ++c)
      {
        dot += static_cast<double>(queries[i * d + c]) *
          static_cast<double>(keys[j * d + c]);
      }
      scores[j] = dot * scale;
    }

    // P_i = softmax(S_i), safe form.
    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    vector<double> weights(n);
    for (int j {0}; j < n; ++j)
    {
      weights[j] = exp(scores[j] - max_score);
      sum += weights[j];
    }

    // O_i = Σ_j P_ij v_j
    for (int a {0}; a < d; ++a)
    {
      double accumulated {0.0};
      for (int j {0}; j < n; ++j)
      {
        accumulated += (weights[j] / sum) *
          static_cast<double>(values[j * d + a]);
      }
      output[i * d + a] = static_cast<float>(accumulated);
    }
  }
  return output;
}

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_inputs(const int count, const int seed)
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
// If all query rows are 0, every score row is 0, softmax is uniform, and each
// output row is the mean of the value rows — checkable by hand.
//------------------------------------------------------------------------------
TEST(ScaledDotProductAttentionTests, ZeroQueriesGiveMeanOfValueRows)
{
  constexpr int n {4};
  constexpr int kHD {4};

  const vector<float> queries(n * kHD, 0.0f);
  const vector<float> keys {make_inputs(n * kHD, 3)};

  // v_j = (j+1) · [1,1,1,1]; the mean over rows is 2.5 in every component.
  vector<float> values(n * kHD);
  for (int j {0}; j < n; ++j)
  {
    for (int a {0}; a < kHD; ++a)
    {
      values[j * kHD + a] = static_cast<float>(j + 1);
    }
  }

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_scores(n * n);
  Array<float> d_weights(n * n);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  scaled_dot_product_attention<float, kHD>(
    d_output.elements_,
    d_scores.elements_,
    d_weights.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < n * kHD; ++i)
  {
    EXPECT_NEAR(output[i], 2.5f, 1e-5f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Full composition against the CPU reference on inputs large enough to
// exercise strided loops in every stage (n > blockDim in the scores stage's
// key loop; multiple warps per softmax block; kHD < blockDim).
//------------------------------------------------------------------------------
TEST(ScaledDotProductAttentionTests, MatchesCpuReference)
{
  constexpr int n {192};
  constexpr int kHD {64};

  const vector<float> queries {make_inputs(n * kHD, 3)};
  const vector<float> keys {make_inputs(n * kHD, 5)};
  const vector<float> values {make_inputs(n * kHD, 11)};

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_scores(n * n);
  Array<float> d_weights(n * n);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  scaled_dot_product_attention<float, kHD>(
    d_output.elements_,
    d_scores.elements_,
    d_weights.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  const vector<float> expected {
    attention_cpu(queries, keys, values, n, kHD)};
  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 1e-4f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// The attention weight matrix P is row-stochastic: every row lies on the
// probability simplex. Inspect the exposed weights workspace directly —
// the intermediate that FlashAttention never materializes.
//------------------------------------------------------------------------------
TEST(ScaledDotProductAttentionTests, WeightRowsLieOnProbabilitySimplex)
{
  constexpr int n {64};
  constexpr int kHD {32};

  const vector<float> queries {make_inputs(n * kHD, 7)};
  const vector<float> keys {make_inputs(n * kHD, 13)};
  const vector<float> values {make_inputs(n * kHD, 17)};

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_scores(n * n);
  Array<float> d_weights(n * n);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  scaled_dot_product_attention<float, kHD>(
    d_output.elements_,
    d_scores.elements_,
    d_weights.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> weights(n * n);
  d_weights.copy_device_output_to_host(weights);

  for (int i {0}; i < n; ++i)
  {
    float row_sum {0.0f};
    for (int j {0}; j < n; ++j)
    {
      EXPECT_GT(weights[i * n + j], 0.0f)
        << "entry (" << i << ", " << j << ")";
      row_sum += weights[i * n + j];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "row " << i;
  }
}

//------------------------------------------------------------------------------
// Large-magnitude queries and keys produce scores around ±10³. The naive
// softmax would overflow exp; the safe softmax stage must keep every output
// finite. Each output row must also stay inside the convex hull of the value
// rows (here all values in [-1, 1], so outputs must be too).
//------------------------------------------------------------------------------
TEST(ScaledDotProductAttentionTests, NumericallyStableForLargeScores)
{
  constexpr int n {32};
  constexpr int kHD {16};

  vector<float> queries {make_inputs(n * kHD, 3)};
  vector<float> keys {make_inputs(n * kHD, 5)};
  for (int i {0}; i < n * kHD; ++i)
  {
    queries[i] *= 100.0f;
    keys[i] *= 100.0f;
  }
  const vector<float> values {make_inputs(n * kHD, 11)};

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_scores(n * n);
  Array<float> d_weights(n * n);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  scaled_dot_product_attention<float, kHD>(
    d_output.elements_,
    d_scores.elements_,
    d_weights.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    ASSERT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
    // O_i ∈ conv{v_1, ..., v_n} and every value component is in [-1, 1].
    ASSERT_GE(output[i], -1.0f) << "index " << i;
    ASSERT_LE(output[i], 1.0f) << "index " << i;
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
