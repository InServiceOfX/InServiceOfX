#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_forward.h"
#include "Transformer/Attention/scaled_dot_product_attention.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::sqrt;
using std::vector;
using Transformer::Attention::flash_attention;
using Transformer::Attention::scaled_dot_product_attention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// Double-precision CPU reference for Att(Q,K,V) = softmax(QK^⊤/√d_k)V with
// the safe softmax.
//------------------------------------------------------------------------------
vector<float> flash_reference_cpu(
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

    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    vector<double> weights(n);
    for (int j {0}; j < n; ++j)
    {
      weights[j] = exp(scores[j] - max_score);
      sum += weights[j];
    }

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
vector<float> make_flash_inputs(const int count, const int seed)
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
// Runs flash_attention on device for given tile parameters and returns the
// host-side output.
//------------------------------------------------------------------------------
template <int kHD, int kBr, int kBc>
vector<float> run_flash_attention(
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const int n)
{
  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention<float, kHD, kBr, kBc>(
    d_output.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> output(n * kHD);
  d_output.copy_device_output_to_host(output);
  return output;
}

//------------------------------------------------------------------------------
// Single tile pair (n ≤ B_r and n ≤ B_c): the inner loop runs exactly once,
// so the fold is a single merge with the identity. Zero queries make the
// expected output the mean of the value rows, checkable by hand.
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, SingleTileZeroQueriesGiveMeanOfValueRows)
{
  constexpr int n {8};
  constexpr int kHD {16};

  const vector<float> queries(n * kHD, 0.0f);
  const vector<float> keys {make_flash_inputs(n * kHD, 5)};

  vector<float> values(n * kHD);
  for (int j {0}; j < n; ++j)
  {
    for (int a {0}; a < kHD; ++a)
    {
      values[j * kHD + a] = static_cast<float>(j + 1);
    }
  }

  const vector<float> output {
    run_flash_attention<kHD, 32, 32>(queries, keys, values, n)};

  // Mean of rows 1·[1,...,1], ..., 8·[1,...,1] is 4.5 in every component.
  for (int i {0}; i < n * kHD; ++i)
  {
    EXPECT_NEAR(output[i], 4.5f, 1e-5f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Multiple K/V tiles (n > B_c): the fold ⊕_j α(A_j) runs over several tiles.
// Compare against the double-precision CPU reference.
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, MatchesCpuReferenceAcrossMultipleTiles)
{
  constexpr int n {192};
  constexpr int kHD {64};

  const vector<float> queries {make_flash_inputs(n * kHD, 3)};
  const vector<float> keys {make_flash_inputs(n * kHD, 5)};
  const vector<float> values {make_flash_inputs(n * kHD, 11)};

  const vector<float> output {
    run_flash_attention<kHD, 64, 32>(queries, keys, values, n)};

  const vector<float> expected {
    flash_reference_cpu(queries, keys, values, n, kHD)};
  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 1e-4f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// The correctness proposition (see the section on The FlashAttention
// Algorithm in FlashAttention.tex): FlashAttention computes Att(Q,K,V)
// *exactly* — the same map as standard attention, only the IO schedule
// differs. Compare directly against the three-kernel composition on device.
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, MatchesStandardAttentionComposition)
{
  constexpr int n {160};
  constexpr int kHD {32};

  const vector<float> queries {make_flash_inputs(n * kHD, 7)};
  const vector<float> keys {make_flash_inputs(n * kHD, 13)};
  const vector<float> values {make_flash_inputs(n * kHD, 17)};

  const vector<float> flash_output {
    run_flash_attention<kHD, 32, 32>(queries, keys, values, n)};

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

  vector<float> standard_output(n * kHD);
  d_output.copy_device_output_to_host(standard_output);

  // Both are float pipelines of the same map; they differ only in the order
  // of the (associative-up-to-rounding) merges.
  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(flash_output[i], standard_output[i], 1e-5f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Sequence length divisible by neither B_r nor B_c: exercises the zero-fill
// of partial Q tiles and the −∞ score padding of the partial last K/V tile
// (identity elements of the merge monoid contribute nothing).
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, HandlesPartialTiles)
{
  constexpr int n {100};
  constexpr int kHD {32};

  const vector<float> queries {make_flash_inputs(n * kHD, 3)};
  const vector<float> keys {make_flash_inputs(n * kHD, 5)};
  const vector<float> values {make_flash_inputs(n * kHD, 11)};

  const vector<float> output {
    run_flash_attention<kHD, 32, 32>(queries, keys, values, n)};

  const vector<float> expected {
    flash_reference_cpu(queries, keys, values, n, kHD)};
  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 1e-4f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Tile shape must not change the result (the fold is over disjoint subsets
// whose union is always {1,...,n}): different (B_r, B_c) choices agree.
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, TileShapeInvariance)
{
  constexpr int n {96};
  constexpr int kHD {32};

  const vector<float> queries {make_flash_inputs(n * kHD, 19)};
  const vector<float> keys {make_flash_inputs(n * kHD, 23)};
  const vector<float> values {make_flash_inputs(n * kHD, 29)};

  const vector<float> output_a {
    run_flash_attention<kHD, 32, 32>(queries, keys, values, n)};
  const vector<float> output_b {
    run_flash_attention<kHD, 64, 16>(queries, keys, values, n)};
  const vector<float> output_c {
    run_flash_attention<kHD, 96, 48>(queries, keys, values, n)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(output_a[i], output_b[i], 1e-5f) << "index " << i;
    ASSERT_NEAR(output_a[i], output_c[i], 1e-5f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Large-magnitude inputs push scores to ~10³, where a naive softmax
// overflows. The running-max rescaling must keep every output finite and
// inside the convex hull of the value rows (all components in [-1, 1]).
//------------------------------------------------------------------------------
TEST(FlashAttentionForwardTests, NumericallyStableForLargeScores)
{
  constexpr int n {64};
  constexpr int kHD {16};

  vector<float> queries {make_flash_inputs(n * kHD, 3)};
  vector<float> keys {make_flash_inputs(n * kHD, 5)};
  for (int i {0}; i < n * kHD; ++i)
  {
    queries[i] *= 100.0f;
    keys[i] *= 100.0f;
  }
  const vector<float> values {make_flash_inputs(n * kHD, 11)};

  const vector<float> output {
    run_flash_attention<kHD, 32, 32>(queries, keys, values, n)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    ASSERT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
    ASSERT_GE(output[i], -1.0f) << "index " << i;
    ASSERT_LE(output[i], 1.0f) << "index " << i;
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
