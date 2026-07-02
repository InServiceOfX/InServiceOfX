#include "DataStructures/Array.h"
#include "Utilities/Memory/streaming_load.h"
#include "gtest/gtest.h"

#include <vector>

using DataStructures::Array;
using std::vector;
using Utilities::Memory::streaming_load;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Memory
{

template <typename T>
__global__ void apply_streaming_load(T* output, const T* input, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    output[idx] = streaming_load<T>(input + idx);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StreamingLoadTests, FloatReadsCorrectValue)
{
  constexpr int N {32};
  const float value {3.14159f};

  vector<float> h_input(N, value);
  Array<float> d_input(N);
  Array<float> d_output(N);
  d_input.copy_host_input_to_device(h_input);

  apply_streaming_load<float><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], value);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StreamingLoadTests, DoubleReadsCorrectValue)
{
  constexpr int N {32};
  const double value {2.718281828459045};

  vector<double> h_input(N, value);
  Array<double> d_input(N);
  Array<double> d_output(N);
  d_input.copy_host_input_to_device(h_input);

  apply_streaming_load<double><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<double> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], value);
  }
}

//------------------------------------------------------------------------------
// Values with exact binary representations: integers that are powers of 2,
// and fractions that are sums of negative powers of 2 (0.5 = 2^-1,
// 0.125 = 2^-3, 0.75 = 2^-1 + 2^-2, etc.). These round-trip through float
// without rounding error, making EXPECT_EQ meaningful beyond just bit identity.
//------------------------------------------------------------------------------
TEST(StreamingLoadTests, FloatExactBinaryRepresentations)
{
  const vector<float> values {
    0.0f,   // zero
    1.0f, 2.0f, 16.0f,        // positive integer powers of 2
    0.5f, 0.25f, 0.125f,      // fractions: 2^-1, 2^-2, 2^-3
    1.5f, 0.75f,               // mixed: 1+2^-1, 2^-1+2^-2
    -1.0f, -0.5f               // negative exact values
  };
  const int N {static_cast<int>(values.size())};

  Array<float> d_input(N);
  Array<float> d_output(N);
  d_input.copy_host_input_to_device(values);

  apply_streaming_load<float><<<1, 32>>>(
    d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], values[i]);
  }
}

} // namespace Memory
} // namespace Utilities
} // namespace GoogleUnitTests
