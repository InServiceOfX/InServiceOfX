#include "DataStructures/Array.h"

#include "gtest/gtest.h"
#include <cuda_fp16.h>  // For half precision support
#include <limits>
#include <vector>

using DataStructures::Array;
using std::numeric_limits;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerics
{
namespace Constants
{

// Do the include here to avoid multiple definitions linking error.
#include "Numerics/Constants/get_infinity.h"

using Numerics::Constants::get_infinity;

template <typename FPType>
__global__ void apply_infinity_kernel(FPType* d_output)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  d_output[index] = get_infinity<FPType>();
}

__global__ void apply_macro(__half* d_output)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  d_output[index] = CUDART_INF_FP16;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetInfinityTests, ReturnsFloat)
{
  Array<float> array {4};
  apply_infinity_kernel<float><<<1, 4>>>(array.elements_);
  vector<float> host_output (4);
  array.copy_device_output_to_host(host_output);
  EXPECT_EQ(host_output.at(0), numeric_limits<float>::infinity());
  EXPECT_EQ(host_output.at(1), numeric_limits<float>::infinity());
  EXPECT_EQ(host_output.at(2), numeric_limits<float>::infinity());
  EXPECT_EQ(host_output.at(3), numeric_limits<float>::infinity());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetInfinityTests, ReturnsDouble)
{
  Array<double> array {4};
  const dim3 blocks {1};
  const dim3 threads {4};
  apply_infinity_kernel<double><<<blocks, threads>>>(array.elements_);
  vector<double> host_output (4);
  array.copy_device_output_to_host(host_output);
  EXPECT_EQ(host_output.at(0), numeric_limits<double>::infinity());
  EXPECT_EQ(host_output.at(1), numeric_limits<double>::infinity());
  EXPECT_EQ(host_output.at(2), numeric_limits<double>::infinity());
  EXPECT_EQ(host_output.at(3), numeric_limits<double>::infinity());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetInfinityTests, ReturnsHalf)
{
  Array<__half> array {4};
  const dim3 blocks {1};
  const dim3 threads {4};
  apply_infinity_kernel<__half><<<blocks, threads>>>(array.elements_);
  vector<__half> host_output (4);
  array.copy_device_output_to_host(host_output);

  EXPECT_EQ(static_cast<__half>(0), numeric_limits<__half>::infinity());

  EXPECT_EQ(host_output.at(0), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(1), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(2), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(3), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetInfinityTests, ApplyMacroComparison)
{
  Array<__half> array {4};
  const dim3 blocks {1};
  const dim3 threads {4};
  apply_macro<<<blocks, threads>>>(array.elements_);
  vector<__half> host_output (4);
  array.copy_device_output_to_host(host_output);
  EXPECT_EQ(host_output.at(0), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(1), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(2), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
  EXPECT_EQ(host_output.at(3), static_cast<__half>(0x7FFFFFFFFFFFFFFF));
}

} // namespace Constants
} // namespace Numerics
} // namespace GoogleUnitTests