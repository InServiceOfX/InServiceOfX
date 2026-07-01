#include "DataStructures/Array.h"
#include "Utilities/Memory/streaming_store.h"
#include "gtest/gtest.h"

#include <vector>

using DataStructures::Array;
using std::vector;
using Utilities::Memory::streaming_store;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Memory
{

template <typename T>
__global__ void apply_streaming_store(T* output, const T value, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    streaming_store(output + idx, value);
  }
}

template <typename T>
__global__ void apply_streaming_store_array(
  T* output, const T* input, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    streaming_store(output + idx, input[idx]);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(StreamingStoreTests, FloatWritesCorrectValue)
{
  constexpr int N {32};
  const float value {3.14159f};

  Array<float> d_output(N);
  apply_streaming_store<float><<<1, 32>>>(d_output.elements_, value, N);
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
TEST(StreamingStoreTests, DoubleWritesCorrectValue)
{
  constexpr int N {32};
  const double value {2.718281828459045};

  Array<double> d_output(N);
  apply_streaming_store<double><<<1, 32>>>(d_output.elements_, value, N);
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
TEST(StreamingStoreTests, FloatExactBinaryRepresentations)
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

  apply_streaming_store_array<float><<<1, 32>>>(
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
