#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_forward.h"
#include "gtest/gtest.h"

#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_multihead_inputs(const int count, const int seed)
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
// Heads act independently (see the section on Multi-Head Attention in
// FlashAttention.tex): a batched launch over B·NH slices must produce, in
// each slice, exactly what a single-slice launch on that slice produces.
//------------------------------------------------------------------------------
TEST(MultiheadFlashAttentionTests, BatchedLaunchMatchesPerSliceLaunches)
{
  constexpr int batch_heads {6};
  constexpr int n {96};
  constexpr int kHD {32};
  constexpr int slice {n * kHD};

  const vector<float> queries {make_multihead_inputs(batch_heads * slice, 3)};
  const vector<float> keys {make_multihead_inputs(batch_heads * slice, 5)};
  const vector<float> values {make_multihead_inputs(batch_heads * slice, 11)};

  // Batched: one launch, gridDim.y = batch_heads.
  Array<float> d_queries(batch_heads * slice);
  Array<float> d_keys(batch_heads * slice);
  Array<float> d_values(batch_heads * slice);
  Array<float> d_output(batch_heads * slice);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention<float, kHD, 32, 32>(
    d_output.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n,
    batch_heads);
  cudaDeviceSynchronize();

  vector<float> batched_output(batch_heads * slice);
  d_output.copy_device_output_to_host(batched_output);

  // Per-slice: batch_heads separate single-head launches on each slice.
  for (int s {0}; s < batch_heads; ++s)
  {
    const vector<float> slice_queries(
      queries.begin() + s * slice, queries.begin() + (s + 1) * slice);
    const vector<float> slice_keys(
      keys.begin() + s * slice, keys.begin() + (s + 1) * slice);
    const vector<float> slice_values(
      values.begin() + s * slice, values.begin() + (s + 1) * slice);

    Array<float> d_slice_queries(slice);
    Array<float> d_slice_keys(slice);
    Array<float> d_slice_values(slice);
    Array<float> d_slice_output(slice);
    d_slice_queries.copy_host_input_to_device(slice_queries);
    d_slice_keys.copy_host_input_to_device(slice_keys);
    d_slice_values.copy_host_input_to_device(slice_values);

    flash_attention<float, kHD, 32, 32>(
      d_slice_output.elements_,
      d_slice_queries.elements_,
      d_slice_keys.elements_,
      d_slice_values.elements_,
      n);
    cudaDeviceSynchronize();

    vector<float> slice_output(slice);
    d_slice_output.copy_device_output_to_host(slice_output);

    // Identical inputs, identical launch geometry per slice: bit-identical.
    for (int i {0}; i < slice; ++i)
    {
      ASSERT_EQ(batched_output[s * slice + i], slice_output[i])
        << "slice " << s << " index " << i;
    }
  }
}

//------------------------------------------------------------------------------
// Causal masking and batching compose: every slice of a causal batched
// launch obeys causality (output row 0 = v_0 of its own slice).
//------------------------------------------------------------------------------
TEST(MultiheadFlashAttentionTests, CausalBatchedFirstRowPerSlice)
{
  constexpr int batch_heads {4};
  constexpr int n {64};
  constexpr int kHD {16};
  constexpr int slice {n * kHD};

  const vector<float> queries {make_multihead_inputs(batch_heads * slice, 7)};
  const vector<float> keys {make_multihead_inputs(batch_heads * slice, 13)};
  const vector<float> values {make_multihead_inputs(batch_heads * slice, 17)};

  Array<float> d_queries(batch_heads * slice);
  Array<float> d_keys(batch_heads * slice);
  Array<float> d_values(batch_heads * slice);
  Array<float> d_output(batch_heads * slice);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);

  flash_attention<float, kHD, 32, 32, true>(
    d_output.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n,
    batch_heads);
  cudaDeviceSynchronize();

  vector<float> output(batch_heads * slice);
  d_output.copy_device_output_to_host(output);

  for (int s {0}; s < batch_heads; ++s)
  {
    for (int a {0}; a < kHD; ++a)
    {
      EXPECT_NEAR(output[s * slice + a], values[s * slice + a], 1e-5f)
        << "slice " << s << " dim " << a;
    }
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
