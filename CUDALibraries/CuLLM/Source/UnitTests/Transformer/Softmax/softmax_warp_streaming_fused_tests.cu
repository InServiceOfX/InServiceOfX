#include "DataStructures/Array.h"
#include "Transformer/Softmax/softmax_warp_streaming_fused.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;
using Transformer::Softmax::softmax_warp_streaming_fused;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Softmax
{

// Launch configuration identical to softmax_warp_fold_reduce: one warp (32
// threads) per block, one block per row. C < 32 is valid: threads with
// rank >= C hold the identity (-inf, 0) and contribute nothing.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxWarpStreamingFusedTests, SingleRowBasic)
{
  constexpr int N {1};
  constexpr int C {4};

  const vector<float> input {1.0f, 2.0f, 3.0f, 4.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_streaming_fused<float><<<N, 32>>>(
    d_output.elements_, d_input.elements_, N, C);
  cudaDeviceSynchronize();

  d_output.copy_device_output_to_host(output);

  const float max_val {4.0f};
  const float e0 {exp(1.0f - max_val)};
  const float e1 {exp(2.0f - max_val)};
  const float e2 {exp(3.0f - max_val)};
  const float e3 {exp(4.0f - max_val)};
  const float sum {e0 + e1 + e2 + e3};

  EXPECT_NEAR(output[0], e0 / sum, 1e-5f);
  EXPECT_NEAR(output[1], e1 / sum, 1e-5f);
  EXPECT_NEAR(output[2], e2 / sum, 1e-5f);
  EXPECT_NEAR(output[3], e3 / sum, 1e-5f);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxWarpStreamingFusedTests, RowSumsToOne)
{
  // C = 64: each of the 32 threads processes 2 elements (thread coarsening).
  constexpr int N {4};
  constexpr int C {64};

  vector<float> input(N * C);
  for (int i {0}; i < N * C; ++i)
  {
    input[i] = static_cast<float>(i % C) * 0.1f;
  }
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_streaming_fused<float><<<N, 32>>>(
    d_output.elements_, d_input.elements_, N, C);
  cudaDeviceSynchronize();

  d_output.copy_device_output_to_host(output);

  for (int row {0}; row < N; ++row)
  {
    float row_sum {0.0f};
    for (int j {0}; j < C; ++j)
    {
      row_sum += output[row * C + j];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "row " << row;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxWarpStreamingFusedTests, NumericalStabilityLargeValues)
{
  // Without the safe-softmax max subtraction, exp(400) overflows to inf.
  constexpr int N {1};
  constexpr int C {4};

  const vector<float> input {100.0f, 200.0f, 300.0f, 400.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_streaming_fused<float><<<N, 32>>>(
    d_output.elements_, d_input.elements_, N, C);
  cudaDeviceSynchronize();

  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < N * C; ++i)
  {
    EXPECT_FALSE(std::isnan(output[i])) << "NaN at index " << i;
    EXPECT_FALSE(std::isinf(output[i])) << "Inf at index " << i;
  }
  // The largest element (400) dominates: softmax output ≈ 1 at index 3.
  EXPECT_NEAR(output[3], 1.0f, 1e-5f);
}

//------------------------------------------------------------------------------
// C not a multiple of 32: exercises the identity-padding lanes in the fold
// (Pass 1) and the strided loop bound in the streaming normalize (Pass 2).
//------------------------------------------------------------------------------
TEST(SoftmaxWarpStreamingFusedTests, RowLengthNotMultipleOfWarpSize)
{
  constexpr int N {3};
  constexpr int C {50};

  vector<float> input(N * C);
  for (int i {0}; i < N * C; ++i)
  {
    input[i] = static_cast<float>((i * 7) % 23 - 11) / 11.0f;
  }
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_streaming_fused<float><<<N, 32>>>(
    d_output.elements_, d_input.elements_, N, C);
  cudaDeviceSynchronize();

  d_output.copy_device_output_to_host(output);

  for (int row {0}; row < N; ++row)
  {
    double max_val {-1e30};
    for (int j {0}; j < C; ++j)
    {
      max_val = std::max(max_val, static_cast<double>(input[row * C + j]));
    }
    double sum {0.0};
    for (int j {0}; j < C; ++j)
    {
      sum += std::exp(static_cast<double>(input[row * C + j]) - max_val);
    }
    for (int j {0}; j < C; ++j)
    {
      const double expected {
        std::exp(static_cast<double>(input[row * C + j]) - max_val) / sum};
      EXPECT_NEAR(output[row * C + j], static_cast<float>(expected), 1e-5f)
        << "row " << row << " col " << j;
    }
  }
}

} // namespace Softmax
} // namespace Transformer
} // namespace GoogleUnitTests
