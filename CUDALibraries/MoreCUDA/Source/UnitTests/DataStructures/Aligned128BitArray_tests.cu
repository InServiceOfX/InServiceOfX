#include "DataStructures/Aligned128BitArray.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using DataStructures::Aligned128BitArray;
using DataStructures::Array;
using DataStructures::load_with_cache_streaming_hint;
using DataStructures::store_to_address;
using std::vector;

namespace GoogleUnitTests
{

namespace DataStructures
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Aligned128BitArray, SizeGetsValueFromTemplateParameter)
{
  // See this for vector types.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=int4#char-short-int-long-longlong-float-double
  // 7.3.1. char, short, int, long, longlong, float, double
  EXPECT_EQ(Aligned128BitArray<int>::size_, 4);
  EXPECT_EQ(Aligned128BitArray<int4>::size_, 1);
  EXPECT_EQ(Aligned128BitArray<uint4>::size_, 1);
  EXPECT_EQ(Aligned128BitArray<float1>::size_, 4);
  EXPECT_EQ(Aligned128BitArray<float2>::size_, 2);
  EXPECT_EQ(Aligned128BitArray<float3>::size_, 1);
  EXPECT_EQ(Aligned128BitArray<float4>::size_, 1);
  EXPECT_EQ(Aligned128BitArray<double1>::size_, 2);
  EXPECT_EQ(Aligned128BitArray<double2>::size_, 1);
  EXPECT_EQ(Aligned128BitArray<double3>::size_, 0);
  EXPECT_EQ(Aligned128BitArray<double4>::size_, 0);
}

template<typename T>
__global__ void test_aligned_array(T* output, const int4 input_value)
{
  Aligned128BitArray<T> aligned_array {input_value};
    
  // Copy values to output
  for (uint32_t i {0}; i < Aligned128BitArray<T>::size_; ++i)
  {
    output[i] = aligned_array[i];
  }
}

template<typename T>
vector<T> run_test_aligned_array(const int4 input_value)
{
  vector<T> output_host (Aligned128BitArray<T>::size_);
  Array<T> output_device {Aligned128BitArray<T>::size_};
    
  test_aligned_array<T><<<1, 1>>>(output_device.elements_, input_value);
        
  output_device.copy_device_output_to_host(output_host);

  return output_host;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Aligned128BitArrayTests, ConstructorWorksWithDifferentTypes) 
{
  // 0x3f800000 is 1.0f in IEEE 754
  // 0x40000000 is 2.0f in IEEE 754
  // 0xbf800000 is -1.0f in IEEE 754
  const int4 test_value {make_int4(0x3f800000, 0x40000000, 0, 0xbf800000)};

  {
    vector<float> result {run_test_aligned_array<float>(test_value)};
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], -1.0f);
  }    
  {
    vector<int> result {run_test_aligned_array<int>(test_value)};
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 0x3f800000);
    EXPECT_EQ(result[1], 0x40000000);
    EXPECT_EQ(result[2], 0);
    EXPECT_EQ(result[3], 0xbf800000);
  }
  {
    vector<float4> result {run_test_aligned_array<float4>(test_value)};
    EXPECT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].x, 1.0f);
    EXPECT_FLOAT_EQ(result[0].y, 2.0f);
    EXPECT_FLOAT_EQ(result[0].z, 0.0f);
    EXPECT_FLOAT_EQ(result[0].w, -1.0f);
  }
}

template<typename T>
__global__ void test_get_as_bits(int4* output, const int4 input_value)
{
  Aligned128BitArray<T> aligned_array {input_value};
  output[0] = aligned_array.get_as_bits();
}

template<typename T>
int4 run_test_get_as_bits(const int4 input_value)
{
  Array<int4> output_device {1};
  
  test_get_as_bits<T><<<1, 1>>>(output_device.elements_, input_value);
  
  vector<int4> temp(1);
  output_device.copy_device_output_to_host(temp);
  return temp[0];
}

TEST(Aligned128BitArrayTests, GetAsBitsWorks) 
{
  const int4 test_value {make_int4(0x3f800000, 0x40000000, 0, 0xbf800000)};
  
  {
    // Test with float type
    const int4 result {run_test_get_as_bits<float>(test_value)};
    EXPECT_EQ(result.x, 0x3f800000u);
    EXPECT_EQ(result.y, 0x40000000u);
    EXPECT_EQ(result.z, 0u);
    EXPECT_EQ(result.w, 0xbf800000u);
  }
  {
    const int4 result {run_test_get_as_bits<int>(test_value)};
    EXPECT_EQ(result.x, 0x3f800000u);
    EXPECT_EQ(result.y, 0x40000000u);
    EXPECT_EQ(result.z, 0u);
    EXPECT_EQ(result.w, 0xbf800000u);
  }  
  {
    const int4 result {run_test_get_as_bits<float4>(test_value)};
    EXPECT_EQ(result.x, 0x3f800000u);
    EXPECT_EQ(result.y, 0x40000000u);
    EXPECT_EQ(result.z, 0u);
    EXPECT_EQ(result.w, 0xbf800000u);
  }
}

template<typename T>
__global__ void test_cache_streaming_load(
  T* output,
  const T* input,
  const uint32_t offset = 0)
{
  auto aligned_array = load_with_cache_streaming_hint<T>(input + offset);
    
  // Copy values to output
  for (uint32_t i{0}; i < Aligned128BitArray<T>::size_; ++i)
  {
    output[i] = aligned_array[i];
  }
}

template<typename T>
vector<T> run_cache_streaming_test(
  const vector<T>& input,
  const uint32_t offset = 0)
{
  vector<T> output_host(Aligned128BitArray<T>::size_);
  Array<T> input_device{static_cast<uint32_t>(input.size())};
  Array<T> output_device{Aligned128BitArray<T>::size_};
    
  input_device.copy_host_input_to_device(input);
    
  test_cache_streaming_load<T><<<1, 1>>>(
    output_device.elements_,
    input_device.elements_,
    offset);
        
  output_device.copy_device_output_to_host(output_host);
    
  return output_host;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Aligned128BitArrayTests, CacheStreamingLoadWorks) 
{
  // Using same IEEE 754 test values as in:
  vector<float> input{1.0f, 2.0f, 0.0f, -1.0f};
  
  {
    vector<float> result{run_cache_streaming_test<float>(input)};
    EXPECT_EQ(result.size(), 4);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], -1.0f);
  }
  {
    vector<float4> vec4_input(1);
    vec4_input[0] = make_float4(1.0f, 2.0f, 0.0f, -1.0f);
    vector<float4> result{run_cache_streaming_test<float4>(vec4_input)};
    EXPECT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].x, 1.0f);
    EXPECT_FLOAT_EQ(result[0].y, 2.0f);
    EXPECT_FLOAT_EQ(result[0].z, 0.0f);
    EXPECT_FLOAT_EQ(result[0].w, -1.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Aligned128BitArrayTests, CacheStreamingLoadWorksWithOffset) 
{
  // Create a larger input array with known patterns
  // 4 aligned chunks
  vector<float> input(16);
  for (uint32_t i = 0; i < input.size(); ++i)
  {
    input[i] = static_cast<float>(i);
  }
    
  // Test loading from different offsets
  for (uint32_t offset {0}; offset < 12; offset += 4)
  {
    vector<float> result{run_cache_streaming_test<float>(input, offset)};
    EXPECT_EQ(result.size(), 4);
    for (uint32_t i {0}; i < 4; ++i)
    {
      EXPECT_FLOAT_EQ(result[i], static_cast<float>(i + offset));
    }
  }
    
  // Test with float4
  vector<float4> vec4_input(4);
  for (uint32_t i {0}; i < 4; ++i)
  {
    vec4_input[i] = make_float4(
      static_cast<float>(i*4), 
      static_cast<float>(i*4 + 1),
      static_cast<float>(i*4 + 2), 
      static_cast<float>(i*4 + 3));
  }
    
  for (uint32_t offset {0}; offset < 3; ++offset)
  {
    vector<float4> result{run_cache_streaming_test<float4>(vec4_input, offset)};
    EXPECT_EQ(result.size(), 1);
    EXPECT_FLOAT_EQ(result[0].x, static_cast<float>(offset*4));
    EXPECT_FLOAT_EQ(result[0].y, static_cast<float>(offset*4 + 1));
    EXPECT_FLOAT_EQ(result[0].z, static_cast<float>(offset*4 + 2));
    EXPECT_FLOAT_EQ(result[0].w, static_cast<float>(offset*4 + 3));
  }
}

// Test for aligned memory access
template<typename T>
__global__ void test_store_to_address_alignment(
  T* output, 
  const int4 input_value, 
  const uint32_t offset)
{
  Aligned128BitArray<T> aligned_array{input_value};
  store_to_address(output + offset, aligned_array);
}

template<typename T>
vector<T> run_test_store_to_address_with_offset(
  const int4 input_value, 
  const uint32_t offset)
{
  // Allocate extra space for offset
  const uint32_t total_size {Aligned128BitArray<T>::size_ + offset};
  Array<T> output_device{total_size};

  test_store_to_address_alignment<T><<<1, 1>>>(
    output_device.elements_, 
    input_value, 
    offset);

  vector<T> output_host(total_size);
  output_device.copy_device_output_to_host(output_host);
    
  return output_host;
}

TEST(Aligned128BitArrayTests, StoreToAddressHandlesAlignment)
{
  const int4 test_value{make_int4(0x3f800000, 0x40000000, 0, 0xbf800000)};
    
  {
    // Test with offset that maintains alignment
    const vector<float> result {
      run_test_store_to_address_with_offset<float>(test_value, 4)};
    EXPECT_EQ(result.size(), 8); // 4 for offset + 4 for data
    EXPECT_FLOAT_EQ(result[4], 1.0f);
    EXPECT_FLOAT_EQ(result[5], 2.0f);
    EXPECT_FLOAT_EQ(result[6], 0.0f);
    EXPECT_FLOAT_EQ(result[7], -1.0f);
  }
  {
    // Test with float4 (should be 16-byte aligned)
    const vector<float4> result{
      run_test_store_to_address_with_offset<float4>(test_value, 1)};
    EXPECT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result[1].x, 1.0f);
    EXPECT_FLOAT_EQ(result[1].y, 2.0f);
    EXPECT_FLOAT_EQ(result[1].z, 0.0f);
    EXPECT_FLOAT_EQ(result[1].w, -1.0f);
  }
}

} // namespace DataStructures
} // namespace GoogleUnitTests
