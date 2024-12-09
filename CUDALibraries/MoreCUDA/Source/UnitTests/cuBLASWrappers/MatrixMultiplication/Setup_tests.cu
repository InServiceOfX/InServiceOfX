#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cublasLt.h>
#include <optional>
#include <tuple>

using std::make_tuple;

using cuBLASWrappers::LibraryContextHandle;
// Use full name for Setup because there is a testing::Test::Setup

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, DefaultConstructs)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, ConstructsWithDimensions)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {16, 15, 14};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, SetupCanSetTransposeOnA)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {4, 3, 2};
  LibraryContextHandle handle {};

  EXPECT_TRUE(setup.setup(handle, true));

  const auto is_transpose_on_A {
    setup.set_descriptor_attributes_.get_transpose_operation_on_A(
      setup.descriptor_.descriptor_)};

  EXPECT_TRUE(is_transpose_on_A);
  EXPECT_EQ(is_transpose_on_A->first, static_cast<int32_t>(CUBLAS_OP_T));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, SetupCanSetBatchCountAndStridedOffsets)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {4, 3, 2};
  LibraryContextHandle handle {};

  ::cuBLASWrappers::MatrixMultiplication::Setup<float>::BatchCountAndStridedOffsets
    batch_count_and_strided_offsets {2};

  batch_count_and_strided_offsets.set_strided_offsets(4, 3, 2);

  EXPECT_TRUE(
    setup.setup(handle, false, false, batch_count_and_strided_offsets));

  EXPECT_EQ(setup.layouts_.get_batch_count('A'), make_tuple(2, 4));
  EXPECT_EQ(setup.layouts_.get_batch_count('B'), make_tuple(2, 4));
  EXPECT_EQ(setup.layouts_.get_batch_count('C'), make_tuple(2, 4));
  EXPECT_EQ(setup.layouts_.get_batch_count('D'), make_tuple(2, 4));

  EXPECT_EQ(setup.layouts_.get_strided_batch_offset('A'), make_tuple(8, 8));
  EXPECT_EQ(setup.layouts_.get_strided_batch_offset('B'), make_tuple(6, 8));
  EXPECT_EQ(setup.layouts_.get_strided_batch_offset('C'), make_tuple(12, 8));
  EXPECT_EQ(setup.layouts_.get_strided_batch_offset('D'), make_tuple(12, 8));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, SetupWorksWithBias)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {3, 4, 2};
  LibraryContextHandle handle {};

  DataStructures::Array<float> bias {
    static_cast<uint32_t>(
      setup.set_descriptor_attributes_.get_bias_size(3, 4))};

  EXPECT_TRUE(
    setup.setup(handle, false, false, std::nullopt, bias.elements_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, SetupWorksWithBiasAndBatchCountAndStridedOffsets)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {3, 4, 2};
  LibraryContextHandle handle {};

  ::cuBLASWrappers::MatrixMultiplication::Setup<float>::BatchCountAndStridedOffsets
    batch_count_and_strided_offsets {2};

  batch_count_and_strided_offsets.set_strided_offsets(3, 4, 2);

  DataStructures::Array<float> bias {
    static_cast<uint32_t>(
      setup.set_descriptor_attributes_.get_bias_size(3, 4))};

  EXPECT_TRUE(
    setup.setup(
      handle,
      false,
      false,
      batch_count_and_strided_offsets,
      bias.elements_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, SetupWithGELUWorks)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {8, 3, 2};
  LibraryContextHandle handle {};

  DataStructures::Array<float> pre_gelu_array {8 * 3};

  EXPECT_TRUE(setup.setup_with_gelu(handle, pre_gelu_array.elements_));
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests