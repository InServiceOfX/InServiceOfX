#include "DataStructures/Array.h"
#include "Utilities/BuiltInTypes/vector_at.h"

#include "gtest/gtest.h"
#include <vector>

using DataStructures::Array;
using Utilities::BuiltInTypes::vector_at;
using std::vector;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace BuiltInTypes
{

template <typename FPType4, typename FPType>
__global__ void apply_vector_at(FPType* d_output, const FPType* d_input)
{
  const unsigned int index {blockIdx.x * blockDim.x + threadIdx.x};
  const FPType4* input_vector {reinterpret_cast<const FPType4*>(d_input)};

  unsigned int index_by_4 {index / 4};

  FPType4 v {input_vector[index_by_4]};

  FPType output {0.0f};

  for (unsigned int i {0}; i < 4; ++i)
  {
    output += vector_at<FPType4, FPType>(v, i);
  }

  d_output[index_by_4] = output;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VectorAtTests, ReturnsFloat)
{
  Array<float> array {32};
  vector<float> host_input (32);
  for (unsigned int i {0}; i < 32; ++i)
  {
    host_input[i] = static_cast<float>(i + 1);
  }
  array.copy_host_input_to_device(host_input);
  Array<float> output {8};

  apply_vector_at<float4, float><<<1, 32>>>(output.elements_, array.elements_);

  vector<float> host_output (8);
  output.copy_device_output_to_host(host_output);

  EXPECT_EQ(host_output.at(0), 10.0f);
  EXPECT_EQ(host_output.at(1), 26.0f);
  EXPECT_EQ(host_output.at(2), 42.0f);
  EXPECT_EQ(host_output.at(3), 58.0f);
  EXPECT_EQ(host_output.at(4), 74.0f);
  EXPECT_EQ(host_output.at(5), 90.0f);
  EXPECT_EQ(host_output.at(6), 106.0f);
  EXPECT_EQ(host_output.at(7), 122.0f);
}

} // namespace BuiltInTypes
} // namespace Utilities
} // namespace GoogleUnitTests