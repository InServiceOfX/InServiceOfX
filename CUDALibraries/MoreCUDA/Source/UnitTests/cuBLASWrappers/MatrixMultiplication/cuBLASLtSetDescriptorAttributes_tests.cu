#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtSetDescriptorAttributes.h"

#include "gtest/gtest.h"

using cuBLASWrappers::MatrixMultiplication::cuBLASLtDescriptor;
using cuBLASWrappers::MatrixMultiplication::cuBLASLtSetDescriptorAttributes;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtSetDescriptorAttributesTests, DefaultConstructs)
{
  cuBLASLtSetDescriptorAttributes set_descriptor_attributes {};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtSetDescriptorAttributesTests, SetTransposeOnAWorks)
{
  cuBLASLtDescriptor descriptor {};

  cuBLASLtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtSetDescriptorAttributesTests, SetTransposeOnBWorks)
{
  cuBLASLtDescriptor descriptor {};

  cuBLASLtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
}

} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
