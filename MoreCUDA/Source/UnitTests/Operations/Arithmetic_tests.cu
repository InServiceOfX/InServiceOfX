#include "DataStructures/Array.h"
#include "Operations/Arithmetic.h"
#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

using DataStructures::Array;
using Operations::add_scalar;
using Operations::add_scalar_2D;
using Operations::addition;
using std::size_t;
using std::vector;
using Utilities::arange;

namespace GoogleUnitTests
{
namespace Operations
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AddScalarAdds)
{
  const size_t example_size {4};

  const vector<float> host_x {0., 1., 2., 3.};
  const size_t threads_per_block {example_size};
  const size_t blocks_per_grid {1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_x);

  add_scalar<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    4.0);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);
  EXPECT_EQ(host_y.at(0), 4.);
  EXPECT_EQ(host_y.at(1), 5.);
  EXPECT_EQ(host_y.at(2), 6.);
  EXPECT_EQ(host_y.at(3), 7.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 1: Map. Implement a "kernel" (GPU function) that adds 10 to each
/// position of vector a and stores it in vector out. You have 1 thread per
/// position.
/// This is the GPU version of that Python code/solution.
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AddScalarAddsWithLvalueScalar)
{
  const size_t example_size {4};
  const vector<float> host_x {0., 1., 2., 3.};
  const size_t threads_per_block {example_size};
  const size_t blocks_per_grid {1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_x);

  const float scalar {10.0};

  add_scalar<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    scalar);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);
  EXPECT_EQ(host_y.at(0), 10.);
  EXPECT_EQ(host_y.at(1), 11.);
  EXPECT_EQ(host_y.at(2), 12.);
  EXPECT_EQ(host_y.at(3), 13.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 2: Zip. Implement a kernel that adds together each position of a and
/// b and stores it in out. You have 1 thread per position.
/// This is the GPU version of that Python code/solution.
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AdditionDoesBinaryAddition)
{
  const size_t example_size {4};
  const auto host_input = arange<float>(example_size);
  const size_t threads_per_block {example_size};
  const size_t blocks_per_grid {1};

  Array<float> addend1 {example_size};
  Array<float> addend2 {example_size};
  Array<float> output {example_size};
  addend1.copy_host_input_to_device(host_input);
  addend2.copy_host_input_to_device(host_input);

  addition<float><<<blocks_per_grid, threads_per_block>>>(
    addend1.elements_,
    addend1.elements_,
    output.elements_);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);

  EXPECT_EQ(host_y.at(0), 0.);
  EXPECT_EQ(host_y.at(1), 2.);
  EXPECT_EQ(host_y.at(2), 4.);
  EXPECT_EQ(host_y.at(3), 6.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 4 - Map 2D: Implement a kernel that adds 10 to each position of a and
/// stores it in out. Input a is 2D and square. You have more threads than
/// positions.
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AddScalar2DAdds)
{
  const size_t example_size {2};
  const auto host_x = arange<float>(4);
  const dim3 threads_per_block {3, 3};
  const size_t blocks_per_grid {1};
  // This works as well.
  //const dim3 blocks_per_grid {1, 1};

  Array<float> input {example_size * example_size};
  Array<float> output {example_size * example_size};
  input.copy_host_input_to_device(host_x);

  add_scalar_2D<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size,
    example_size,
    10.0);

  vector<float> host_y (example_size * example_size);
  output.copy_device_output_to_host(host_y);

  EXPECT_EQ(host_y.at(0), 10.);
  EXPECT_EQ(host_y.at(1), 11.);
  EXPECT_EQ(host_y.at(2), 12.);
  EXPECT_EQ(host_y.at(3), 13.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 6: Blocks. Implement a kernel that adds 10 to each position of a and
/// stores it in out. You have fewer threads per block than the size of a.
/// This is the GPU version of that Python code/solution.
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AddScalarAddsWithLessThreads)
{
  const size_t example_size {9};
  const auto host_input = arange<float>(example_size);
  const dim3 threads_per_block {4, 1};
  const dim3 blocks_per_grid {3, 1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_input);

  add_scalar<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size,
    10.0);

  vector<float> host_y (example_size, 0.0);
  output.copy_device_output_to_host(host_y);

  EXPECT_EQ(host_y.at(0), 10.);
  EXPECT_EQ(host_y.at(1), 11.);
  EXPECT_EQ(host_y.at(2), 12.);
  EXPECT_EQ(host_y.at(3), 13.);
  EXPECT_EQ(host_y.at(4), 14.);
  EXPECT_EQ(host_y.at(5), 15.);
  EXPECT_EQ(host_y.at(6), 16.);
  EXPECT_EQ(host_y.at(7), 17.);
  EXPECT_EQ(host_y.at(8), 18.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 7 - Blocks 2D. Implement the same kernel in 2D. You have fewer
/// threads per block than the size of a in both directions.
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AddScalar2DAddsWithFewerThreads)
{
  const size_t example_size {5};
  const vector<float> host_input (example_size * example_size, 1.0);
  const dim3 threads_per_block {3, 3};
  const dim3 blocks_per_grid {2, 2};

  Array<float> input {example_size * example_size};
  Array<float> output {example_size * example_size};
  input.copy_host_input_to_device(host_input);

  add_scalar_2D<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size,
    example_size,
    10.0);

  vector<float> host_y (example_size * example_size, 0.0);
  output.copy_device_output_to_host(host_y);

  for (auto i {0}; i < example_size * example_size; ++i)
  {
    EXPECT_FLOAT_EQ(host_y.at(i), 11.0);
  }
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArithmeticTests, AdditionChecksSize)
{
  const size_t example_size {4};
  const vector<float> host_x {0., 1., 2., 3.};
  const size_t threads_per_block {8};
  // This works as well.
  const size_t blocks_per_grid {1};

  Array<float> input1 {example_size};
  Array<float> input2 {example_size};
  Array<float> output {example_size};
  input1.copy_host_input_to_device(host_x);
  input2.copy_host_input_to_device(host_x);

  addition<float><<<blocks_per_grid, threads_per_block>>>(
    input1.elements_,
    input2.elements_,
    output.elements_,
    example_size);

  vector<float> host_y (example_size);
  output.copy_device_output_to_host(host_y);

  EXPECT_EQ(host_y.at(0), 0.);
  EXPECT_EQ(host_y.at(1), 2.);
  EXPECT_EQ(host_y.at(2), 4.);
  EXPECT_EQ(host_y.at(3), 6.);
}

} // namespace Operations
} // namespace GoogleUnitTests