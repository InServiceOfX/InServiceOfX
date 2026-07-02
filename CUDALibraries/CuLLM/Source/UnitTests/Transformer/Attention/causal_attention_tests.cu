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
// Double-precision CPU reference for causal attention: row i's softmax is
// restricted to keys j ≤ i (the causal mask M_ij = −∞ for j > i makes the
// weight row a distribution supported on {1, ..., i}).
//------------------------------------------------------------------------------
vector<float> causal_attention_cpu(
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
    // Only unmasked scores j ≤ i participate.
    vector<double> scores(i + 1);
    for (int j {0}; j <= i; ++j)
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
    vector<double> weights(i + 1);
    for (int j {0}; j <= i; ++j)
    {
      weights[j] = exp(scores[j] - max_score);
      sum += weights[j];
    }

    for (int a {0}; a < d; ++a)
    {
      double accumulated {0.0};
      for (int j {0}; j <= i; ++j)
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
vector<float> make_causal_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

template <int kHD, int kBr, int kBc>
vector<float> run_causal_flash(
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

  flash_attention<float, kHD, kBr, kBc, true>(
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

template <int kHD>
vector<float> run_causal_standard(
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const int n)
{
  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_scores(n * n);
  Array<float> d_weights(n * n);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  scaled_dot_product_attention<float, kHD, true>(
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
  return output;
}

//------------------------------------------------------------------------------
// Row 0 may attend only to key 0: its weight row is the point mass at
// position 0, so output row 0 is exactly v_0 — the sharpest hand-checkable
// consequence of the mask.
//------------------------------------------------------------------------------
TEST(CausalAttentionTests, FirstRowEqualsFirstValueRow)
{
  constexpr int n {64};
  constexpr int kHD {32};

  const vector<float> queries {make_causal_inputs(n * kHD, 3)};
  const vector<float> keys {make_causal_inputs(n * kHD, 5)};
  const vector<float> values {make_causal_inputs(n * kHD, 11)};

  const vector<float> flash_output {
    run_causal_flash<kHD, 32, 32>(queries, keys, values, n)};
  const vector<float> standard_output {
    run_causal_standard<kHD>(queries, keys, values, n)};

  for (int a {0}; a < kHD; ++a)
  {
    EXPECT_NEAR(flash_output[a], values[a], 1e-5f) << "dim " << a;
    EXPECT_NEAR(standard_output[a], values[a], 1e-5f) << "dim " << a;
  }
}

//------------------------------------------------------------------------------
// Causality property: output row i must not depend on key/value rows j > i.
// Overwrite all future rows with garbage and check row i is unchanged.
//------------------------------------------------------------------------------
TEST(CausalAttentionTests, OutputIndependentOfFuturePositions)
{
  constexpr int n {48};
  constexpr int kHD {16};
  constexpr int probe_row {20};

  const vector<float> queries {make_causal_inputs(n * kHD, 3)};
  vector<float> keys {make_causal_inputs(n * kHD, 5)};
  vector<float> values {make_causal_inputs(n * kHD, 11)};

  const vector<float> baseline {
    run_causal_flash<kHD, 16, 16>(queries, keys, values, n)};

  // Corrupt every key and value row strictly after probe_row.
  for (int j {probe_row + 1}; j < n; ++j)
  {
    for (int a {0}; a < kHD; ++a)
    {
      keys[j * kHD + a] = 777.0f;
      values[j * kHD + a] = -777.0f;
    }
  }

  const vector<float> corrupted {
    run_causal_flash<kHD, 16, 16>(queries, keys, values, n)};

  // Rows 0..probe_row see only uncorrupted keys/values: bit-identical work.
  for (int i {0}; i <= probe_row; ++i)
  {
    for (int a {0}; a < kHD; ++a)
    {
      ASSERT_EQ(baseline[i * kHD + a], corrupted[i * kHD + a])
        << "row " << i << " dim " << a;
    }
  }
}

//------------------------------------------------------------------------------
// Full causal map against the CPU reference, with n not a multiple of the
// tile sizes so diagonal-straddling and partial tiles both occur.
//------------------------------------------------------------------------------
TEST(CausalAttentionTests, FlashMatchesCpuReference)
{
  constexpr int n {100};
  constexpr int kHD {32};

  const vector<float> queries {make_causal_inputs(n * kHD, 7)};
  const vector<float> keys {make_causal_inputs(n * kHD, 13)};
  const vector<float> values {make_causal_inputs(n * kHD, 17)};

  const vector<float> output {
    run_causal_flash<kHD, 32, 32>(queries, keys, values, n)};
  const vector<float> expected {
    causal_attention_cpu(queries, keys, values, n, kHD)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(output[i], expected[i], 1e-4f) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Flash and the three-kernel composition compute the same causal map; the
// composition realizes the mask as −∞ scores through the safe softmax while
// flash skips tiles above the diagonal — the results must still agree.
//------------------------------------------------------------------------------
TEST(CausalAttentionTests, FlashMatchesStandardComposition)
{
  constexpr int n {160};
  constexpr int kHD {64};

  const vector<float> queries {make_causal_inputs(n * kHD, 19)};
  const vector<float> keys {make_causal_inputs(n * kHD, 23)};
  const vector<float> values {make_causal_inputs(n * kHD, 29)};

  const vector<float> flash_output {
    run_causal_flash<kHD, 64, 32>(queries, keys, values, n)};
  const vector<float> standard_output {
    run_causal_standard<kHD>(queries, keys, values, n)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(flash_output[i], standard_output[i], 1e-5f) << "index " << i;
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
