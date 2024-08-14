#include "DataStructures/Array.h"
#include "Operations/dot_product.h"
#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>

using DataStructures::Array;
using Operations::dot_product;
using std::size_t;
using std::vector;
using std::transform;
using Utilities::arange;

namespace GoogleUnitTests
{
namespace Operations
{

template <typename T>
T sum_of_squares(const size_t N)
{
  const T n {static_cast<T>(N)};
  return n * (n + 1) * (2 * n + 1) / 6;
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 10 - Dot Product
/// Implement a kernel that computes the dot-product of a and b and stores it
/// in out. You have 1 thread per position. You only need 2 global reads and 1
/// global write per thread.
//------------------------------------------------------------------------------
TEST(DotProductTests, DotProductWorks)
{
  const size_t example_size {8};
  const auto host_input = arange<float>(example_size);
  constexpr dim3 threads_per_block {example_size, 1};
  const dim3 blocks_per_grid {1, 1};

  Array<float> input1 {example_size};
  Array<float> input2 {example_size};
  Array<float> output {1};
  input1.copy_host_input_to_device(host_input);
  input2.copy_host_input_to_device(host_input);

  dot_product<threads_per_block.x, float>
    <<<blocks_per_grid, threads_per_block>>>(
      input1.elements_,
      input2.elements_,
      output.elements_,
      example_size);

  vector<float> host_y (1, 0.0);

  output.copy_device_output_to_host(host_y);

  //  1 + 4 + 9 + 16 + 25 + 36 + 49 = 140
  EXPECT_EQ(host_y.at(0), 140);
}

TEST(DotProductTests, DotProductWorksOnLargerValues)
{
  constexpr size_t N {33 * 1024};
  constexpr size_t threads_per_block {256};
  constexpr size_t blocks_per_grid {
    (N + threads_per_block - 1) / threads_per_block};

  const auto host_input = arange<float>(N);
  auto host_input_2 = host_input;

  transform(
    host_input_2.begin(),
    host_input_2.end(),
    host_input_2.begin(),
    [](float x) { return 2 * x; });

  Array<float> input1 {N};
  Array<float> input2 {N};
  Array<float> output {1};
  input1.copy_host_input_to_device(host_input);
  input2.copy_host_input_to_device(host_input_2);  // Use the squared values

  dot_product<threads_per_block, float>
    <<<blocks_per_grid, threads_per_block>>>(
      input1.elements_,
      input2.elements_,
      output.elements_,
      N);

  vector<float> host_y (1, 0.0);

  output.copy_device_output_to_host(host_y);

  // Calculate the expected result using the sum of squares formula
  const float expected {2 * sum_of_squares<float>(N - 1)};

  EXPECT_FLOAT_EQ(host_y.at(0), expected);
}

} // namespace Operations
} // namespace GoogleUnitTests