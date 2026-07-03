#if defined(CULLM_HAS_CUTLASS)

#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_cute.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention_cute;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// CuTe smoke test: layout algebra evaluates at compile/host time exactly as
// documented — proves the vendored CUTLASS headers compile in this build
// before any kernel-level test can blame them.
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, CuteLayoutAlgebraSmoke)
{
  using namespace cute;
  // Row-major (16, 64): element (r, c) at r*64 + c.
  auto layout {make_layout(
    make_shape(Int<16>{}, Int<64>{}),
    make_stride(Int<64>{}, Int<1>{}))};
  EXPECT_EQ(size(layout), 16 * 64);
  EXPECT_EQ(layout(0, 0), 0);
  EXPECT_EQ(layout(1, 0), 64);
  EXPECT_EQ(layout(0, 1), 1);
  EXPECT_EQ(layout(3, 7), 3 * 64 + 7);
}

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-0.5, 0.5], pre-rounded to
// half so the CPU reference sees exactly what the kernel sees.
//------------------------------------------------------------------------------
vector<float> make_cute_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    const float value {
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 210.0f};
    result[i] = __half2float(__float2half(value));
  }
  return result;
}

//------------------------------------------------------------------------------
// Double-precision CPU attention reference over one (batch, head) slice.
//------------------------------------------------------------------------------
void cute_attention_cpu(
  vector<float>& output,
  vector<float>& logsumexp,
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const int sequence_length,
  const int head_dim,
  const bool causal)
{
  const double scale {1.0 / std::sqrt(static_cast<double>(head_dim))};
  output.assign(static_cast<size_t>(sequence_length) * head_dim, 0.0f);
  logsumexp.assign(sequence_length, 0.0f);

  for (int i {0}; i < sequence_length; ++i)
  {
    const int limit {causal ? i + 1 : sequence_length};
    vector<double> scores(limit);
    for (int j {0}; j < limit; ++j)
    {
      double dot {0.0};
      for (int d {0}; d < head_dim; ++d)
      {
        dot += static_cast<double>(queries[i * head_dim + d]) *
          static_cast<double>(keys[j * head_dim + d]);
      }
      scores[j] = dot * scale;
    }
    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    for (int j {0}; j < limit; ++j)
    {
      sum += std::exp(scores[j] - max_score);
    }
    logsumexp[i] = static_cast<float>(max_score + std::log(sum));
    for (int d {0}; d < head_dim; ++d)
    {
      double accumulated {0.0};
      for (int j {0}; j < limit; ++j)
      {
        accumulated += (std::exp(scores[j] - max_score) / sum) *
          static_cast<double>(values[j * head_dim + d]);
      }
      output[static_cast<size_t>(i) * head_dim + d] =
        static_cast<float>(accumulated);
    }
  }
}

//------------------------------------------------------------------------------
// Device harness mirroring flash_attention_tensor_core_tests: run the CuTe
// kernel over number_of_batch_heads slices and compare every slice against
// the CPU reference at fp16-scale tolerance (P rounds to half before P·V).
// kHeadDim is fixed at 64 and kWarpsPerBlock at 4 by the kernel's cp.async
// copy shape.
//------------------------------------------------------------------------------
template <bool kCausal>
void run_cute_and_compare(
  const int sequence_length,
  const int number_of_batch_heads,
  const bool check_logsumexp)
{
  constexpr int kHD {64};
  const int slice_elements {sequence_length * kHD};
  const int total_elements {number_of_batch_heads * slice_elements};

  const vector<float> queries {make_cute_inputs(total_elements, 3)};
  const vector<float> keys {make_cute_inputs(total_elements, 5)};
  const vector<float> values {make_cute_inputs(total_elements, 11)};

  const auto to_half {[](const vector<float>& x)
  {
    vector<__half> result(x.size());
    for (size_t i {0}; i < x.size(); ++i)
    {
      result[i] = __float2half(x[i]);
    }
    return result;
  }};

  Array<__half> d_queries(total_elements);
  Array<__half> d_keys(total_elements);
  Array<__half> d_values(total_elements);
  Array<__half> d_output(total_elements);
  Array<__half> d_logsumexp(number_of_batch_heads * sequence_length);
  {
    vector<__half> h {to_half(queries)};
    d_queries.copy_host_input_to_device(h);
  }
  {
    vector<__half> h {to_half(keys)};
    d_keys.copy_host_input_to_device(h);
  }
  {
    vector<__half> h {to_half(values)};
    d_values.copy_host_input_to_device(h);
  }

  flash_attention_cute<__half, kHD, 4, kCausal>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    sequence_length,
    number_of_batch_heads);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  vector<__half> output(total_elements);
  vector<__half> logsumexp(number_of_batch_heads * sequence_length);
  d_output.copy_device_output_to_host(output);
  d_logsumexp.copy_device_output_to_host(logsumexp);

  for (int slice {0}; slice < number_of_batch_heads; ++slice)
  {
    const vector<float> q_slice(
      queries.begin() + slice * slice_elements,
      queries.begin() + (slice + 1) * slice_elements);
    const vector<float> k_slice(
      keys.begin() + slice * slice_elements,
      keys.begin() + (slice + 1) * slice_elements);
    const vector<float> v_slice(
      values.begin() + slice * slice_elements,
      values.begin() + (slice + 1) * slice_elements);

    vector<float> expected;
    vector<float> expected_logsumexp;
    cute_attention_cpu(
      expected, expected_logsumexp, q_slice, k_slice, v_slice,
      sequence_length, kHD, kCausal);

    for (int i {0}; i < slice_elements; ++i)
    {
      ASSERT_NEAR(
        __half2float(output[slice * slice_elements + i]),
        expected[i],
        1.5e-2f)
        << "slice " << slice << " element " << i;
    }
    if (check_logsumexp)
    {
      for (int i {0}; i < sequence_length; ++i)
      {
        ASSERT_NEAR(
          __half2float(logsumexp[slice * sequence_length + i]),
          expected_logsumexp[i],
          2e-2f)
          << "slice " << slice << " row " << i;
      }
    }
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, MatchesCpuReferenceTileMultiple)
{
  run_cute_and_compare<false>(128, 1, true);
}

//------------------------------------------------------------------------------
// Ragged sequence exercises the scalar-fallback tail-tile copy path and
// every mask guard.
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, RaggedSequenceLength)
{
  run_cute_and_compare<false>(100, 1, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, CausalMatchesCpuReference)
{
  run_cute_and_compare<true>(128, 1, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, CausalRaggedSequenceLength)
{
  run_cute_and_compare<true>(90, 1, false);
}

//------------------------------------------------------------------------------
// Multiple (batch, head) slices via gridDim.y must stay independent, and
// long-enough sequences exercise many double-buffer iterations.
//------------------------------------------------------------------------------
TEST(FlashAttentionCuteTests, MultiSliceIndependence)
{
  run_cute_and_compare<false>(192, 6, true);
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests

#endif // CULLM_HAS_CUTLASS
