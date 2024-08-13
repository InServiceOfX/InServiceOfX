//------------------------------------------------------------------------------
/// \details Some CUDA C++ solutions to GPU-Puzzles by Sasha Rush. See
/// \ref https://github.com/srush/GPU-Puzzles
/// \ref https://github.com/isamu-isozaki/GPU-Puzzles-answers
/// and search other unit tests for the CUDA C++ solutions to puzzles not shown
/// here.
//------------------------------------------------------------------------------

#include "DataStructures/Array.h"
#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

using DataStructures::Array;
using std::size_t;
using std::vector;
using Utilities::arange;

template <typename T>
__global__ void add_with_guard(
  T* input,
  T* output,
  const size_t size,
  const T scalar)
{
  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  if (index < size)
  {
    output[index] = input[index] + scalar;
  }
}

template <typename T>
__global__ void broadcast_addition(
  T* addend1,
  T* addend2,
  T* output,
  const size_t size)
{
  const std::size_t i {blockIdx.x * blockDim.x + threadIdx.x};
  const std::size_t j {blockIdx.y * blockDim.y + threadIdx.y};

  if (i < size && j < size)
  {
    const std::size_t index {i + j * size};

    output[index] = addend1[i] + addend2[j];
  }
}

//------------------------------------------------------------------------------
/// This does not work:
/// template <typename T>
/// __global__ void add_scalar_with_shared_memory(
///   T* input,
///   T* output,
///   const std::size_t size,
///   const T scalar,
///   const std::size_t threads_per_block)
/// As Rush said, "Each block can only have a constant amount of shared memory"
/// and so it can't be dynamically declared, otherwise:
/// error: a variable length array cannot have static storage duration
//------------------------------------------------------------------------------

template <std::size_t THREADS_PER_BLOCK, typename T>
__global__ void add_scalar_with_shared_memory(
  T* input,
  T* output,
  const std::size_t size,
  const T scalar)
{
  __shared__ T shared_array[THREADS_PER_BLOCK];

  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  if (index < size)
  {
    shared_array[threadIdx.x] = input[index];

    __syncthreads();
  }

  if (index < size)
  {
    output[index] = shared_array[threadIdx.x] + scalar;
  }
}

//------------------------------------------------------------------------------
/// \details Assume P <= THREADS_PER_BLOCK.
//------------------------------------------------------------------------------
template <std::size_t THREADS_PER_BLOCK, std::size_t P, typename T>
__global__ void pooling_with_shared_memory(
  T* input,
  T* output,
  const std::size_t size)
{
  __shared__ T shared_array[P - 1 + THREADS_PER_BLOCK];

  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  if (index < size)
  {
    shared_array[threadIdx.x + P - 1] = input[index];
  }
  if (threadIdx.x < P - 1)
  {
    shared_array[threadIdx.x] = (index < P - 1) ? static_cast<T>(0) :
      input[index - (P - 1)];
  }

  __syncthreads();

  if (index < size)
  {
    for (auto i {0}; i < P; ++i)
    {
      output[index] += shared_array[threadIdx.x + P - 1 - i];
    }
  }
}

namespace GoogleUnitTests
{

//------------------------------------------------------------------------------
/// Puzzle 3 - Guards
/// Implement a kernel that adds 10 to each position of a and stores it in out.
/// You have more threads than positions.
//------------------------------------------------------------------------------
TEST(GPUPuzzlesTests, Guards)
{
  const size_t example_size {4};
  const vector<float> host_x {0., 1., 2., 3.};
  const size_t threads_per_block {8};
  const size_t blocks_per_grid {1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_x);

  add_with_guard<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size,
    10.0);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);
  EXPECT_EQ(host_y.at(0), 10.);
  EXPECT_EQ(host_y.at(1), 11.);
  EXPECT_EQ(host_y.at(2), 12.);
  EXPECT_EQ(host_y.at(3), 13.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 5 - Broadcast: Implement a kernel that adds a and b and stores it in
/// out. Inputs a and b are vectors. You have more threads than positions.
//------------------------------------------------------------------------------
TEST(GPUPuzzlesTests, BroadcastAdds)
{
  const size_t example_size {2};
  const vector<float> host_x {0., 1.};
  const dim3 threads_per_block {3, 3};
  const size_t blocks_per_grid {1};

  Array<float> input1 {example_size};
  Array<float> input2 {example_size};
  Array<float> output {example_size * example_size};
  input1.copy_host_input_to_device(host_x);
  input2.copy_host_input_to_device(host_x);

  broadcast_addition<float><<<blocks_per_grid, threads_per_block>>>(
    input1.elements_,
    input2.elements_,
    output.elements_,
    example_size);

  vector<float> host_y (example_size * example_size);

  output.copy_device_output_to_host(host_y);
  EXPECT_EQ(host_y.at(0), 0.);
  EXPECT_EQ(host_y.at(1), 1.);
  EXPECT_EQ(host_y.at(2), 1.);
  EXPECT_EQ(host_y.at(3), 2.);
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 8 - Shared: Implement a kernel that adds 10 to each position of a and
/// stores it in out. You have fewer threads per block than the size of a.
/// From Rush:
/// Warning: Each block can only have a constant amount of shared memory that
/// threads in that block can read and write to. This needs to be a literal
/// python constant not a variable. After writing to shared memory you need to
/// call cuda.syncthreads to ensure that threads do not cross.
///
/// (This example does not really need shared memory or syncthreads, but it is a
/// demo.)
//------------------------------------------------------------------------------
TEST(GPUPuzzlesTests, AddScalarWithSharedMemory)
{
  const size_t example_size {8};
  const vector<float> host_x (example_size, 1.);
  // If you do
  // const dim3 threads_per_block {4, 1};
  // then in add_scalar_with_shared_memory<..>, in template parameters, it will
  // error:
  // error: expression must have a constant value
  // note #2701-D: attempt to access run-time storage
  constexpr dim3 threads_per_block {4, 1};
  const dim3 blocks_per_grid {2, 1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_x);

  add_scalar_with_shared_memory<threads_per_block.x, float>
    <<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size,
    10.0);
 
  vector<float> host_y (example_size);
  output.copy_device_output_to_host(host_y);

  for (auto i = 0; i < example_size; ++i)
  {
    EXPECT_FLOAT_EQ(host_y.at(i), 11.0);
  }
}

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 9 - Pooling: Implement a kernel that sums together the last 3
/// position of a and stores it in out. You have 1 thread per position. You only
/// need 1 global read and 1 global write per thread.
//------------------------------------------------------------------------------
TEST(GPUPuzzlesTests, PoolingWorksFor3)
{
  const size_t example_size {8};
  const auto host_input = arange<float>(example_size);
  constexpr dim3 threads_per_block {8, 1};
  const dim3 blocks_per_grid {1, 1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_input);

  pooling_with_shared_memory<threads_per_block.x, 3, float>
    <<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    example_size);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);

  EXPECT_EQ(host_y.at(0), 0.);
  EXPECT_EQ(host_y.at(1), 1.);
  EXPECT_EQ(host_y.at(2), 3.);
  EXPECT_EQ(host_y.at(3), 6.);
  EXPECT_EQ(host_y.at(4), 9.);
  EXPECT_EQ(host_y.at(5), 12.);
  EXPECT_EQ(host_y.at(6), 15.);
  EXPECT_EQ(host_y.at(7), 18.);
}

} // namespace GoogleUnitTests