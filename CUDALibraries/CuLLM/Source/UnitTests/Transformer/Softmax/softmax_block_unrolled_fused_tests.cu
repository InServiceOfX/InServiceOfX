#include "DataStructures/Array.h"
#include "Transformer/Softmax/softmax_block_unrolled_fused.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;
using Transformer::Softmax::softmax_block_unrolled_fused;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Softmax
{

// Launch: softmax_block_unrolled_fused<T, kUnrollFactor><<<N, block_size, shared_bytes>>>
//   shared_bytes = 2 * (block_size / 32) * sizeof(AccT)
//   block_size must be a multiple of 32.
//
// Threads with tid >= C skip all three loops (loop condition i < C is false).
// They still participate in warp reductions, contributing identity values
// (-infinity for max, 0.0 for sum). No early exit is used — all threads
// reach block.sync() naturally, so the barrier is always complete.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxBlockUnrolledFusedTests, SingleRowBasic)
{
  // N=1, C=4, block_size=32, kUnrollFactor=8 (default).
  // C < block_size: only threads 0-3 have valid indices.
  // Each iteration of the outer loop (i = tid) tries to read 8 chunks of
  // block_width=32; all but the first chunk of thread 0,1,2,3 are OOB
  // (clamped to x[3]) and not written. Verifies correctness for small C.
  constexpr int N {1};
  constexpr int C {4};
  constexpr int block_size {32};

  const vector<float> input {1.0f, 2.0f, 3.0f, 4.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_block_unrolled_fused<float><<<N, block_size,
    2 * (block_size / 32) * sizeof(float)>>>(
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
TEST(SoftmaxBlockUnrolledFusedTests, LargeRowSumsToOne)
{
  // N=4, C=1024, block_size=128, kUnrollFactor=8 (default).
  // Outer loop stride = 128 * 8 = 1024 = C, so each thread visits exactly
  // kUnrollFactor elements per outer iteration. Exercises the full unrolled
  // path with no OOB reads (C is an exact multiple of stride).
  constexpr int N {4};
  constexpr int C {1024};
  constexpr int block_size {128};

  vector<float> input(N * C);
  for (int i {0}; i < N * C; ++i)
  {
    input[i] = static_cast<float>(i % C) * 0.01f;
  }
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_block_unrolled_fused<float><<<N, block_size,
    2 * (block_size / 32) * sizeof(float)>>>(
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
    EXPECT_NEAR(row_sum, 1.0f, 1e-4f) << "row " << row;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxBlockUnrolledFusedTests, NumericalStabilityLargeValues)
{
  // Without max subtraction, exp(400) overflows to inf.
  constexpr int N {1};
  constexpr int C {4};
  constexpr int block_size {32};

  const vector<float> input {100.0f, 200.0f, 300.0f, 400.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_block_unrolled_fused<float><<<N, block_size,
    2 * (block_size / 32) * sizeof(float)>>>(
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

} // namespace Softmax
} // namespace Transformer
} // namespace GoogleUnitTests
