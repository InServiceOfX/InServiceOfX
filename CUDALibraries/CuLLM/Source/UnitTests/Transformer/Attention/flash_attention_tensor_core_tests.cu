#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_tensor_core.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::Attention::flash_attention_tensor_core;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// WMMA fragments require tensor cores (sm_70+); this suite additionally
// assumes the fp16 16x16x16 shapes used by the kernel. The Check binary is
// compiled for sm_75/sm_86 only, and CUDA enumerates devices fastest-first,
// so device 0 is the RTX 3060 on this machine — this test makes that
// visible and guards against a surprise device change.
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, RunsOnTensorCoreCapableDevice)
{
  int device {-1};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp properties {};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  std::printf(
    "Running on CUDA device %d: %s (sm_%d%d)\n",
    device, properties.name, properties.major, properties.minor);
  EXPECT_GE(properties.major * 10 + properties.minor, 75)
    << "tensor-core kernel tests require sm_75+ (got " << properties.name
    << ")";
}

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-0.5, 0.5], pre-rounded to
// half so the CPU reference sees exactly what the kernel sees.
//------------------------------------------------------------------------------
vector<float> make_half_inputs(const int count, const int seed)
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
// Double-precision CPU attention reference over one (batch, head) slice,
// with optional logsumexp output.
//------------------------------------------------------------------------------
void attention_cpu(
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
// Device harness: run the tensor-core kernel over number_of_batch_heads
// slices and compare every slice against the CPU reference. P is rounded
// to half before the P·V matmul (as in cuDNN/FlashAttention-2), so the
// tolerance is fp16-scale: 1.5e-2 absolute on outputs of magnitude <= 0.5.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, bool kCausal>
void run_and_compare(
  const int sequence_length,
  const int number_of_batch_heads,
  const bool check_logsumexp)
{
  const int slice_elements {sequence_length * kHD};
  const int total_elements {number_of_batch_heads * slice_elements};

  const vector<float> queries {make_half_inputs(total_elements, 3)};
  const vector<float> keys {make_half_inputs(total_elements, 5)};
  const vector<float> values {make_half_inputs(total_elements, 11)};

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

  flash_attention_tensor_core<__half, kHD, kWarps, kCausal>(
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
    attention_cpu(
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
        // L is stored in half: tolerance is half-ULP at |L| up to ~ln(n).
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
TEST(FlashAttentionTensorCoreTests, MatchesCpuReferenceTileMultiple)
{
  // Sequence a multiple of both the 16-wide K/V tile and the block's
  // 64-row coverage: no ragged edges anywhere.
  run_and_compare<64, 4, false>(128, 1, true);
}

//------------------------------------------------------------------------------
// Ragged sequence exercises every out-of-range guard: partial K/V tile,
// partial query tile, and a warp whose rows are entirely out of range.
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, RaggedSequenceLength)
{
  run_and_compare<64, 4, false>(100, 1, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, CausalMatchesCpuReference)
{
  run_and_compare<64, 4, true>(128, 1, true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, CausalRaggedSequenceLength)
{
  run_and_compare<64, 4, true>(90, 1, false);
}

//------------------------------------------------------------------------------
// Multiple (batch, head) slices via gridDim.y must stay independent.
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, MultiSliceIndependence)
{
  run_and_compare<64, 2, false>(48, 6, true);
}

//------------------------------------------------------------------------------
// kHeadDim = 32 and 48: multiples of the 16-wide WMMA tile that are not
// multiples of 32 or powers of two, exercising the fragment loop bounds.
//------------------------------------------------------------------------------
TEST(FlashAttentionTensorCoreTests, HeadDim32)
{
  run_and_compare<32, 4, false>(80, 2, false);
}

TEST(FlashAttentionTensorCoreTests, HeadDim48)
{
  run_and_compare<48, 2, true>(70, 2, false);
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
