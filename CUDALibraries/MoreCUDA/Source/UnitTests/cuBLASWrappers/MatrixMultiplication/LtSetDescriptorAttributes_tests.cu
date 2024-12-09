#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <cstdint>

using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
using DataStructures::Array;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
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
TEST(LtSetDescriptorAttributesTests, GetTransposeOperationOnAWorks)
{
  LtDescriptor descriptor {};

  LtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));

  const auto transpose_operation_on_A {
    set_descriptor_attributes.get_transpose_operation_on_A(
      descriptor.descriptor_)};

  EXPECT_TRUE(transpose_operation_on_A);
  EXPECT_EQ(transpose_operation_on_A->first, static_cast<int32_t>(CUBLAS_OP_N));
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
TEST(LtSetDescriptorAttributesTests, GetTransposeOperationOnBWorks)
{
  LtDescriptor descriptor {};

  LtSetDescriptorAttributes set_descriptor_attributes {};

  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_,
    true));

  const auto transpose_operation_on_B {
    set_descriptor_attributes.get_transpose_operation_on_B(
      descriptor.descriptor_)};

  EXPECT_TRUE(transpose_operation_on_B);
  EXPECT_EQ(transpose_operation_on_B->first, static_cast<int32_t>(CUBLAS_OP_T));
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtSetDescriptorAttributesTests, SetGELUEpilogueAuxiliaryPointerWorks)
{
  const uint64_t M {10};
  const uint64_t N {11};

  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};

  Array<float> A_array {M * N};

  EXPECT_TRUE(set_descriptor_attributes.set_gelu_epilogue_auxiliary_pointer(
    descriptor.descriptor_, A_array.elements_));
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
