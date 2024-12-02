#include "DataStructures/Array.h"

#include "gtest/gtest.h"
#include <vector>

using DataStructures::Array;
using std::vector;

namespace GoogleUnitTests
{

__global__ void get_built_in_variables_sizes(unsigned int* sizes)
{
  const unsigned int tid {threadIdx.x + blockIdx.x * blockDim.x};

  if (tid == 0)
  {
    // Variable of type dim3
    sizes[0] = sizeof(gridDim);
    // Variable of type uint3
    sizes[1] = sizeof(blockIdx);
    // Variable of type dim3
    sizes[2] = sizeof(blockDim);
    // Variable of type uint3
    sizes[3] = sizeof(threadIdx);
    // Variable of type int
    sizes[4] = sizeof(warpSize);
  }
  else if (tid == 1)
  {
    sizes[5] = sizeof(gridDim.x);
    sizes[6] = sizeof(gridDim.y);
    sizes[7] = sizeof(gridDim.z);
  }
  else if (tid == 2)
  {
    sizes[8] = sizeof(blockIdx.x);
    sizes[9] = sizeof(blockIdx.y);
    sizes[10] = sizeof(blockIdx.z);
  }
  else if (tid == 3)
  {
    sizes[11] = sizeof(blockDim.x);
    sizes[12] = sizeof(blockDim.y);
    sizes[13] = sizeof(blockDim.z);
  }
  else if (tid == 4)
  {
    sizes[14] = sizeof(threadIdx.x);
    sizes[15] = sizeof(threadIdx.y);
    sizes[16] = sizeof(threadIdx.z);
  }

  __syncthreads();
}

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables
//------------------------------------------------------------------------------
TEST(BuiltInVariablesTests, GetSizes)
{
  EXPECT_EQ(sizeof(unsigned int), 4);

  Array<unsigned int> d_sizes(17);
  get_built_in_variables_sizes<<<1, 5>>>(d_sizes.elements_);

  vector<unsigned int> sizes(17);
  d_sizes.copy_device_output_to_host(sizes);

  EXPECT_EQ(sizes[0], 12);
  EXPECT_EQ(sizes[1], 12);
  EXPECT_EQ(sizes[2], 12);
  EXPECT_EQ(sizes[3], 12);
  EXPECT_EQ(sizes[4], 4);

  EXPECT_EQ(sizes[5], 4);
  EXPECT_EQ(sizes[6], 4);
  EXPECT_EQ(sizes[7], 4);

  EXPECT_EQ(sizes[8], 4);
  EXPECT_EQ(sizes[9], 4);
  EXPECT_EQ(sizes[10], 4);

  EXPECT_EQ(sizes[11], 4);
  EXPECT_EQ(sizes[12], 4);
  EXPECT_EQ(sizes[13], 4);

  EXPECT_EQ(sizes[14], 4);
  EXPECT_EQ(sizes[15], 4);
  EXPECT_EQ(sizes[16], 4);
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3
//------------------------------------------------------------------------------
TEST(BuiltInVectorTypes, dim3HasSize)
{
  // Based on uint3. Any component left unspecified is initialized to 1.
  EXPECT_EQ(sizeof(dim3), 12);

  const dim3 example {1, 2};

  EXPECT_EQ(sizeof(example.x), 4);
  EXPECT_EQ(sizeof(example.y), 4);
  EXPECT_EQ(sizeof(example.z), 4);
}

__global__ void get_built_in_vector_alignment_requirements(unsigned int* sizes)
{
  const unsigned int tid {threadIdx.x + blockIdx.x * blockDim.x};

  if (tid == 0)
  {
    sizes[0] = sizeof(char1);
    sizes[1] = sizeof(uchar1);
    sizes[2] = sizeof(char2);
    sizes[3] = sizeof(uchar2);
    sizes[4] = sizeof(char3);
    sizes[5] = sizeof(uchar3);
    sizes[6] = sizeof(char4);
    sizes[7] = sizeof(uchar4);
  }
  else if (tid == 1)
  {
    sizes[8] = sizeof(short1);
    sizes[9] = sizeof(ushort1);
    sizes[10] = sizeof(short2);
    sizes[11] = sizeof(ushort2);
    sizes[12] = sizeof(short3);
    sizes[13] = sizeof(ushort3);
    sizes[14] = sizeof(short4);
    sizes[15] = sizeof(ushort4);
  }
  else if (tid == 2)
  {
    sizes[16] = sizeof(int1);
    sizes[17] = sizeof(uint1);
    sizes[18] = sizeof(int2);
    sizes[19] = sizeof(uint2);
    sizes[20] = sizeof(int3);
    sizes[21] = sizeof(uint3);
    sizes[22] = sizeof(int4);
    sizes[23] = sizeof(uint4);
  }

  __syncthreads();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BuiltInVectorTypes, AlignmentRequirements)
{
  EXPECT_EQ(sizeof(char), 1);
  EXPECT_EQ(sizeof(unsigned char), 1);

  EXPECT_EQ(sizeof(short), 2);
  EXPECT_EQ(sizeof(unsigned short), 2);

  EXPECT_EQ(sizeof(int), 4);
  EXPECT_EQ(sizeof(unsigned int), 4);

  Array<unsigned int> d_sizes(24);
  get_built_in_vector_alignment_requirements<<<1, 3>>>(d_sizes.elements_);

  vector<unsigned int> sizes(24);
  d_sizes.copy_device_output_to_host(sizes);

  EXPECT_EQ(sizes[0], 1);
  EXPECT_EQ(sizes[1], 1);
  EXPECT_EQ(sizes[2], 2);
  EXPECT_EQ(sizes[3], 2);
  EXPECT_EQ(sizes[4], 3);
  EXPECT_EQ(sizes[5], 3);
  EXPECT_EQ(sizes[6], 4);
  EXPECT_EQ(sizes[7], 4);

  EXPECT_EQ(sizes[8], 2);
  EXPECT_EQ(sizes[9], 2);
  EXPECT_EQ(sizes[10], 4);
  EXPECT_EQ(sizes[11], 4);
  EXPECT_EQ(sizes[12], 6);
  EXPECT_EQ(sizes[13], 6);
  EXPECT_EQ(sizes[14], 8);
  EXPECT_EQ(sizes[15], 8);

  EXPECT_EQ(sizes[16], 4);
  EXPECT_EQ(sizes[17], 4);
  EXPECT_EQ(sizes[18], 8);
  EXPECT_EQ(sizes[19], 8);
  EXPECT_EQ(sizes[20], 12);
  EXPECT_EQ(sizes[21], 12);
  EXPECT_EQ(sizes[22], 16);
  EXPECT_EQ(sizes[23], 16);
}

} // namespace GoogleUnitTests

