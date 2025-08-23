#include "Configuration/GPUConfiguration.h"
#include "ParallelProcessing/warp_reductions.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"
#include "Utilities/DeviceManagement/GetAndSetGPUDevices.h"

#include <algorithm>
#include <filesystem>
#include <vector>

using Configuration::GPUConfiguration;
using DataStructures::Array;
using ParallelProcessing::warp_reduce_max;
using ParallelProcessing::warp_reduce_max_shift_up;
using ParallelProcessing::warp_reduce_sum;
using ParallelProcessing::warp_reduce_sum_with_shuffle_down;
using Utilities::DeviceManagement::GetAndSetGPUDevices;
using std::vector;

namespace GoogleUnitTests
{
namespace ParallelProcessing
{

std::filesystem::path get_device_configuration_path()
{
  return std::filesystem::path(__FILE__).parent_path() /
    "../../../Configurations/GPUConfiguration.txt";
}

auto find_max_value_and_position(const std::vector<float>& vec)
{
  if (vec.empty())
  {
    // Return a std::pair with position -1 to indicate invalid result
    return std::make_pair(-1.0f, static_cast<std::size_t>(-1));
  }

  auto max_iter = std::max_element(vec.begin(), vec.end());
  return std::make_pair(
    *max_iter,
    static_cast<std::size_t>(std::distance(vec.begin(), max_iter)));
}

// Helper kernel to test warp_reduce_max
template <typename FPType>
__global__ void test_warp_reduce_max_kernel(
  FPType* output,
  const FPType* input,
  const int size)
{
  const int tid {static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  FPType thread_value {input[tid]};

  // Perform warp-level reduction
  FPType warp_max {warp_reduce_max(thread_value)};

  // Only write result if thread is in bounds
  if (tid < size)
  {
    output[tid] = warp_max;
  }
}

template <typename FPType>
__global__ void test_warp_reduce_max_shift_up_kernel(
  FPType* output,
  const FPType* input,
  const int size)
{
  const int tid {static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  FPType thread_value {input[tid]};

  FPType warp_max {warp_reduce_max_shift_up(thread_value)};

  if (tid < size)
  {
    output[tid] = warp_max;
  }
}

template <typename FPType>
__global__ void test_warp_reduce_sum_kernel(
  FPType* output,
  const FPType* input,
  const int size)
{
  const int tid {static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  FPType thread_value {input[tid]};
  FPType warp_sum {warp_reduce_sum(thread_value)};

  if (tid < size)
  {
    output[tid] = warp_sum;
  }
}

template <typename FPType>
__global__ void test_warp_reduce_sum_with_shuffle_down_kernel(
  FPType* output,
  const FPType* input,
  const int size)
{
  const int tid {static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};

  FPType thread_value {input[tid]};
  FPType warp_sum {warp_reduce_sum_with_shuffle_down(thread_value)};

  if (tid < size)
  {
    output[tid] = warp_sum;
  }
}

TEST(WarpReduceMaxTests, BasicFunctionality)
{
  const int warp_size {32};
  
  // Test case 1: Simple ascending values within a warp
  vector<float> input(warp_size);
  for (int i {0}; i < warp_size; ++i)
  {
    input[i] = static_cast<float>(i);
  }
  
  Array<float> d_input(input.size());
  Array<float> d_output(input.size());
  
  d_input.copy_host_input_to_device(input);
  
  test_warp_reduce_max_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);
  
  vector<float> output(warp_size);
  d_output.copy_device_output_to_host(output);
  
  // All threads in the warp should have the maximum value (31)
  for (int i {0}; i < warp_size; ++i)
  {
    EXPECT_FLOAT_EQ(output[i], 31.0f);
  }

  auto [max_value, max_position] = find_max_value_and_position(output);
  EXPECT_FLOAT_EQ(max_value, 31.0f);
  EXPECT_EQ(max_position, 0);
}

TEST(WarpReduceMaxTests, NegativeValues)
{
  const int warp_size {32};

  {
    // Test case 2: Negative values
    vector<float> input(warp_size, -1.0f);
    input.at(15) = 0.0f;  // Maximum value in middle of warp

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    // Notice that for threads 0 to 15, the max was obtained.
    for (int i {0}; i < warp_size / 2; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
    for (int i {warp_size / 2}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(16) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i <= warp_size / 2; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
    for (int i {warp_size / 2 + 1}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(warp_size - 2) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i < warp_size - 1; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
    for (int i {warp_size - 1}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(warp_size - 1) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
  }
}

TEST(WarpReduceMaxTests, MultipleWarps)
{
  const int num_warps {4};
  const int warp_size {32};
  const int total_threads {num_warps * warp_size};

  // Test case 3: Different max values per warp
  vector<float> input(total_threads, -1.0f);
  for (int w {0}; w < num_warps; ++w)
  {
    input.at(w * warp_size + w) = static_cast<float>(w);
  }

  Array<float> d_input(input.size());
  Array<float> d_output(input.size());

  d_input.copy_host_input_to_device(input);

  test_warp_reduce_max_kernel<<<1, total_threads>>>(
    d_output.elements_,
    d_input.elements_,
    total_threads);

  vector<float> output(total_threads);
  d_output.copy_device_output_to_host(output);

  for (int w {0}; w < num_warps; ++w)
  {
    for (int i {0}; i < warp_size; ++i)
    {
      if (i <= w)
      {
        EXPECT_FLOAT_EQ(output[w * warp_size + i], static_cast<float>(w));
      }
      else
      {
        EXPECT_FLOAT_EQ(output[w * warp_size + i], -1.0f);
      }
    }
  }
}

void test_arbitrary_max_value_position(const std::size_t max_value_position)
{
  static constexpr int warp_size {32};

  vector<float> input(warp_size);
  for (int i {0}; i < warp_size; ++i)
  {
    input[i] = static_cast<float>(i);
  }
  input.at(max_value_position) = 31.1f;

  Array<float> d_input(input.size());
  Array<float> d_output(input.size());
  
  d_input.copy_host_input_to_device(input);

  test_warp_reduce_max_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);
  
  vector<float> output(warp_size);
  d_output.copy_device_output_to_host(output);

  EXPECT_FLOAT_EQ(output.at(0), 31.1f);

  auto [max_value, max_position] = find_max_value_and_position(output);
  EXPECT_FLOAT_EQ(max_value, 31.1f);
  EXPECT_EQ(max_position, 0);
}

TEST(WarpReduceMaxTests, MaxValueAppearsInZerothPosition)
{
  for (std::size_t i {0}; i < 32; ++i)
  {
    test_arbitrary_max_value_position(i);
  }
}

TEST(WarpReduceMaxShiftUpTests, NegativeValues)
{
  const int warp_size {32};

  {
    vector<float> input(warp_size, -1.0f);
    input.at(15) = 0.0f;  // Maximum value in middle of warp

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_shift_up_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    // Notice that for threads 0 to 15, the max was obtained.
    for (int i {0}; i < warp_size / 2 - 1; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
    for (int i {warp_size / 2 - 1}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(16) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_shift_up_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i < warp_size / 2; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
    for (int i {warp_size / 2}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(warp_size - 2) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_shift_up_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i < warp_size - 2; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }
    for (int i {warp_size - 2}; i < warp_size; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), 0.0f);
    }
  }
  {
    vector<float> input(warp_size, -1.0f);
    input.at(warp_size - 1) = 0.0f;

    Array<float> d_input(input.size());
    Array<float> d_output(input.size());

    d_input.copy_host_input_to_device(input);

    test_warp_reduce_max_shift_up_kernel<<<1, warp_size>>>(
      d_output.elements_,
      d_input.elements_,
      warp_size);

    vector<float> output(warp_size);
    d_output.copy_device_output_to_host(output);

    for (int i {0}; i < warp_size - 1; ++i)
    {
      EXPECT_FLOAT_EQ(output.at(i), -1.0f);
    }

    EXPECT_FLOAT_EQ(output.at(warp_size - 1), 0.0f);
  }
}

TEST(WarpReduceSumTests, SumsAcrossWarp)
{
  const int warp_size {32};
  
  vector<float> input(warp_size);
  for (int i {0}; i < warp_size; ++i)
  {
    // This is 1, 2, 3, ... 32.
    input[i] = static_cast<float>(i + 1.0f);
  }
  
  Array<float> d_input(input.size());
  Array<float> d_output(input.size());
  
  d_input.copy_host_input_to_device(input);
  
  test_warp_reduce_sum_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);
  
  vector<float> output(warp_size);
  d_output.copy_device_output_to_host(output);
  
  // Use the sum formula for the first n natural numbers, S = n(n + 1) / 2.
  for (int i {0}; i < warp_size; ++i)
  {
    EXPECT_FLOAT_EQ(output.at(i), (warp_size + 1) * warp_size / 2);
  }

  for (int i {0}; i < warp_size; ++i)
  {
    // This is 0, 1, 2, 0, ... 0.
    if (i < 3)
    {
      input[i] = static_cast<float>(i);
    }
    else
    {
      input[i] = 0.0f;
    }
  }

  d_input.copy_host_input_to_device(input);
  test_warp_reduce_sum_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);

  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < warp_size; ++i)
  {
    EXPECT_FLOAT_EQ(output.at(i), (2 + 1) * 2 / 2);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WarpReduceSumWithShuffleDownTests, SumsAcrossWarpWithShuffleDown)
{
  GPUConfiguration gpu_configuration {};
  ASSERT_TRUE(std::filesystem::exists(get_device_configuration_path()));
  gpu_configuration.parse_configuration_file(get_device_configuration_path());
  GetAndSetGPUDevices gasgd {};
  ASSERT_TRUE(
    gasgd.set_device(gpu_configuration.get_configuration_struct().device_id));

  constexpr int warp_size {32};
  vector<float> input(warp_size);
  for (int i {0}; i < warp_size; ++i)
  {
    input[i] = static_cast<float>(i + 1.0f);
  }

  Array<float> d_input(input.size());
  Array<float> d_output(input.size());
  
  d_input.copy_host_input_to_device(input);

  test_warp_reduce_sum_with_shuffle_down_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);
  
  vector<float> output(warp_size);
  d_output.copy_device_output_to_host(output);
  
  // Use the sum formula for the first n natural numbers, S = n(n + 1) / 2.
  for (int i {0}; i < warp_size; ++i)
  {
    EXPECT_FLOAT_EQ(output.at(i), (warp_size + 1) * warp_size / 2 + 16.0f * i);
  }

  for (int i {0}; i < warp_size; ++i)
  {
    // This is 0, 1, 2, 0, ... 0.
    if (i < 3)
    {
      input[i] = static_cast<float>(i);
    }
    else
    {
      input[i] = 0.0f;
    }
  }

  d_input.copy_host_input_to_device(input);
  test_warp_reduce_sum_with_shuffle_down_kernel<<<1, warp_size>>>(
    d_output.elements_,
    d_input.elements_,
    warp_size);

  d_output.copy_device_output_to_host(output);

  for (int i {0}; i < warp_size; ++i)
  {
    std::cout << "i: " << i << ", output[i]: " << output[i] << std::endl;
  }
}

} // namespace ParallelProcessing
} // namespace GoogleUnitTests
