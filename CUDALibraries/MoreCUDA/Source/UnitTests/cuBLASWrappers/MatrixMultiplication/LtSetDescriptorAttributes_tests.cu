#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"

#include "gtest/gtest.h"

using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, DefaultConstructs)
{
  LtSetDescriptorAttributes set_descriptor_attributes {};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetTransposeOnAWorks)
{
  LtDescriptor descriptor {};

  LtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetTransposeOnBWorks)
{
  LtDescriptor descriptor {};

  LtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetDefaultEpilogueWorks)
{
  LtSetDescriptorAttributes set_descriptor_attributes {};

  set_descriptor_attributes.set_epilogue(false, false, false);

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetDefaultEpilogueFunctionWorks)
{
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};

  set_descriptor_attributes.set_epilogue(false, false, false);

  EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
    descriptor.descriptor_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetDefaultScaleTypeWorks)
{
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_scale_type(descriptor.descriptor_));
}

} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
