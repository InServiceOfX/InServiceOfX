#include "DataStructures/Array.h"
#include "Transformer/Softmax/softmax_warp_fold_reduce.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;
using Transformer::Softmax::softmax_warp_fold_reduce;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Softmax
{

// Launch configuration: one warp (32 threads) per block, one block per row.
// With blockDim.x = 32: meta_group_size()=1, meta_group_rank()=0,
// so row_index = blockIdx.x. Grid = N gives one block per row.
// C < 32 is valid: threads with rank >= C hold the identity (-inf, 0) and
// contribute nothing to the Level 2 reduction.

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SoftmaxWarpFoldReduceTests, SingleRowBasic)
{
  constexpr int N {1};
  constexpr int C {4};

  const vector<float> input {1.0f, 2.0f, 3.0f, 4.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_fold_reduce<float><<<N, 32>>>(
    d_output.elements_, d_input.elements_, N, C);
  cudaDeviceSynchronize();

  d_output.copy_device_output_to_host(output);

  // CPU reference: softmax([1,2,3,4]), max = 4
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
TEST(SoftmaxWarpFoldReduceTests, RowSumsToOne)
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

  softmax_warp_fold_reduce<float><<<N, 32>>>(
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
TEST(SoftmaxWarpFoldReduceTests, NumericalStabilityLargeValues)
{
  // Without the safe-softmax max subtraction, exp(400) overflows to inf.
  constexpr int N {1};
  constexpr int C {4};

  const vector<float> input {100.0f, 200.0f, 300.0f, 400.0f};
  vector<float> output(N * C);

  Array<float> d_input(N * C);
  Array<float> d_output(N * C);
  d_input.copy_host_input_to_device(input);

  softmax_warp_fold_reduce<float><<<N, 32>>>(
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
