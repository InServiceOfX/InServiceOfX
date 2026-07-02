#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_forward.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::log;
using std::sqrt;
using std::vector;
using Transformer::Attention::flash_attention;
using Transformer::Attention::flash_attention_warp_cooperative;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_warp_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

template <int kHD, int kWarps, bool kCausal = false>
void run_warp_cooperative(
  vector<float>& output,
  vector<float>& logsumexp,
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const int n)
{
  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  Array<float> d_logsumexp(n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention_warp_cooperative<float, kHD, kWarps, kCausal>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  output.resize(n * kHD);
  logsumexp.resize(n);
  d_output.copy_device_output_to_host(output);
  d_logsumexp.copy_device_output_to_host(logsumexp);
}

//------------------------------------------------------------------------------
// The execution mapping changed (warp per row, distributed accumulator); the
// map did not. The warp-cooperative kernel must agree with the
// one-thread-per-row kernel, whose correctness is established against CPU
// references — including on partial tiles (n not a multiple of 32).
//------------------------------------------------------------------------------
TEST(FlashAttentionWarpCooperativeTests, MatchesThreadPerRowKernel)
{
  constexpr int kHD {64};

  for (const int n : {8, 100, 192, 500})
  {
    const vector<float> queries {make_warp_inputs(n * kHD, 3)};
    const vector<float> keys {make_warp_inputs(n * kHD, 5)};
    const vector<float> values {make_warp_inputs(n * kHD, 11)};

    vector<float> warp_output;
    vector<float> logsumexp;
    run_warp_cooperative<kHD, 8>(
      warp_output, logsumexp, queries, keys, values, n);

    Array<float> d_queries(n * kHD);
    Array<float> d_keys(n * kHD);
    Array<float> d_values(n * kHD);
    Array<float> d_output(n * kHD);
    d_queries.copy_host_input_to_device(queries);
    d_keys.copy_host_input_to_device(keys);
    d_values.copy_host_input_to_device(values);

    flash_attention<float, kHD, 64, 32>(
      d_output.elements_,
      d_queries.elements_,
      d_keys.elements_,
      d_values.elements_,
      n);
    cudaDeviceSynchronize();

    vector<float> reference_output(n * kHD);
    d_output.copy_device_output_to_host(reference_output);

    for (int i {0}; i < n * kHD; ++i)
    {
      ASSERT_NEAR(warp_output[i], reference_output[i], 1e-5f)
        << "n = " << n << " index " << i;
    }
  }
}

//------------------------------------------------------------------------------
// The logsumexp output is L_i = ln Σ_j exp(s_ij) (in safe form m + ln ℓ).
// Verify against a double-precision CPU computation.
//------------------------------------------------------------------------------
TEST(FlashAttentionWarpCooperativeTests, LogsumexpMatchesCpuReference)
{
  constexpr int n {100};
  constexpr int kHD {32};

  const vector<float> queries {make_warp_inputs(n * kHD, 7)};
  const vector<float> keys {make_warp_inputs(n * kHD, 13)};
  const vector<float> values {make_warp_inputs(n * kHD, 17)};

  vector<float> output;
  vector<float> logsumexp;
  run_warp_cooperative<kHD, 4>(
    output, logsumexp, queries, keys, values, n);

  const double scale {1.0 / sqrt(static_cast<double>(kHD))};
  for (int i {0}; i < n; ++i)
  {
    vector<double> scores(n);
    for (int j {0}; j < n; ++j)
    {
      double dot {0.0};
      for (int d {0}; d < kHD; ++d)
      {
        dot += static_cast<double>(queries[i * kHD + d]) *
          static_cast<double>(keys[j * kHD + d]);
      }
      scores[j] = dot * scale;
    }
    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    for (int j {0}; j < n; ++j)
    {
      sum += exp(scores[j] - max_score);
    }
    const double expected {max_score + log(sum)};

    ASSERT_NEAR(logsumexp[i], static_cast<float>(expected), 1e-4f)
      << "row " << i;
  }
}

//------------------------------------------------------------------------------
// Causal: agrees with the causal one-thread-per-row kernel, including
// diagonal-straddling tiles and n not a multiple of the tile sizes.
//------------------------------------------------------------------------------
TEST(FlashAttentionWarpCooperativeTests, CausalMatchesThreadPerRowKernel)
{
  constexpr int n {150};
  constexpr int kHD {32};

  const vector<float> queries {make_warp_inputs(n * kHD, 19)};
  const vector<float> keys {make_warp_inputs(n * kHD, 23)};
  const vector<float> values {make_warp_inputs(n * kHD, 29)};

  vector<float> warp_output;
  vector<float> logsumexp;
  run_warp_cooperative<kHD, 4, true>(
    warp_output, logsumexp, queries, keys, values, n);

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention<float, kHD, 32, 32, true>(
    d_output.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);
  cudaDeviceSynchronize();

  vector<float> reference_output(n * kHD);
  d_output.copy_device_output_to_host(reference_output);

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(warp_output[i], reference_output[i], 1e-5f) << "index " << i;
  }

  // Causal row 0 attends only to key 0: L_0 = s_00 exactly.
  double dot {0.0};
  for (int d {0}; d < kHD; ++d)
  {
    dot += static_cast<double>(queries[d]) * static_cast<double>(keys[d]);
  }
  EXPECT_NEAR(
    logsumexp[0],
    static_cast<float>(dot / sqrt(static_cast<double>(kHD))),
    1e-5f);
}

//------------------------------------------------------------------------------
// Batched slices are independent: batched launch equals per-slice launches.
//------------------------------------------------------------------------------
TEST(FlashAttentionWarpCooperativeTests, BatchedMatchesPerSlice)
{
  constexpr int batch_heads {4};
  constexpr int n {64};
  constexpr int kHD {32};
  constexpr int slice {n * kHD};

  const vector<float> queries {make_warp_inputs(batch_heads * slice, 3)};
  const vector<float> keys {make_warp_inputs(batch_heads * slice, 5)};
  const vector<float> values {make_warp_inputs(batch_heads * slice, 11)};

  Array<float> d_queries(batch_heads * slice);
  Array<float> d_keys(batch_heads * slice);
  Array<float> d_values(batch_heads * slice);
  Array<float> d_output(batch_heads * slice);
  Array<float> d_logsumexp(batch_heads * n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention_warp_cooperative<float, kHD, 4>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n,
    batch_heads);
  cudaDeviceSynchronize();

  vector<float> batched_output(batch_heads * slice);
  d_output.copy_device_output_to_host(batched_output);

  for (int s {0}; s < batch_heads; ++s)
  {
    const vector<float> slice_queries(
      queries.begin() + s * slice, queries.begin() + (s + 1) * slice);
    const vector<float> slice_keys(
      keys.begin() + s * slice, keys.begin() + (s + 1) * slice);
    const vector<float> slice_values(
      values.begin() + s * slice, values.begin() + (s + 1) * slice);

    vector<float> slice_output;
    vector<float> slice_logsumexp;
    run_warp_cooperative<kHD, 4>(
      slice_output,
      slice_logsumexp,
      slice_queries,
      slice_keys,
      slice_values,
      n);

    for (int i {0}; i < slice; ++i)
    {
      ASSERT_EQ(batched_output[s * slice + i], slice_output[i])
        << "slice " << s << " index " << i;
    }
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
