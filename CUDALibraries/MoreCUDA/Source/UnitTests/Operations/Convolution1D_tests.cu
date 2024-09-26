#include "DataStructures/Array.h"
#include "Operations/Convolution1D.h"
#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

using DataStructures::Array;
using Operations::convolve_1d;
using std::size_t;
using std::vector;
using Utilities::arange;

namespace GoogleUnitTests
{
namespace Operations
{

//------------------------------------------------------------------------------
/// \ref https://github.com/srush/GPU-Puzzles
/// Puzzle 11 - 1D Convolution. Implement a kernel that computes a 1D
/// convolution between a and b and stores it in out. You only need 2 global
/// reads and 1 global write per thread.
//------------------------------------------------------------------------------
TEST(Convolution1DTests, Convolve1DConvolves)
{
  {
    constexpr size_t N_a {6};
    constexpr size_t N_b {3};
    const auto host_input_a = arange<float>(N_a);
    const auto host_input_b = arange<float>(N_b);

    Array<float> input_a {N_a};
    Array<float> input_b {N_b};
    Array<float> output {N_a};
    input_a.copy_host_input_to_device(host_input_a);
    input_b.copy_host_input_to_device(host_input_b);

    constexpr size_t THREADS_PER_BLOCK {256};

    convolve_1d<THREADS_PER_BLOCK, N_b, float>
      <<<1, THREADS_PER_BLOCK>>>(
        output.elements_, input_a.elements_, input_b.elements_, N_a);

    const auto host_expected = vector<float> {5, 8, 11, 14, 5, 0};
    vector<float> host_y (N_a, 0.0);
    output.copy_device_output_to_host(host_y);
    EXPECT_EQ(host_expected, host_y);
  }
  {
    constexpr size_t N_a {15};
    constexpr size_t N_b {4};
    const auto host_input_a = arange<float>(N_a);
    const auto host_input_b = arange<float>(N_b);

    Array<float> input_a {N_a};
    Array<float> input_b {N_b};
    Array<float> output {N_a};
    input_a.copy_host_input_to_device(host_input_a);
    input_b.copy_host_input_to_device(host_input_b);

    constexpr size_t THREADS_PER_BLOCK {256};

    convolve_1d<THREADS_PER_BLOCK, N_b, float>
      <<<1, THREADS_PER_BLOCK>>>(
        output.elements_, input_a.elements_, input_b.elements_, N_a);

    const auto host_expected = vector<float> {
      14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 41, 14, 0};

    vector<float> host_y (N_a, 0.0);
    output.copy_device_output_to_host(host_y);
    EXPECT_EQ(host_expected, host_y);
  }
}

} // namespace Operations
} // namespace GoogleUnitTests