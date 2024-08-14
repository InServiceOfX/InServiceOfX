#include "DataStructures/Array.h"
#include "Operations/dot_product.h"
#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <cuda_runtime.h>

using DataStructures::Array;
using Operations::dot_product;
using std::size_t;
using std::vector;
using Utilities::arange;

namespace GoogleUnitTests
{
namespace Operations
{

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

} // namespace Operations
} // namespace GoogleUnitTests