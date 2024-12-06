#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "Utilities/CaptureCerr.h"

#include "gtest/gtest.h"

#include <string>
#include <tuple>

using cuBLASWrappers::MatrixMultiplication::LtLayouts;
using Utilities::CaptureCerr;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, DefaultConstructor)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {};

    EXPECT_EQ(layouts.A_layout_, nullptr);
    EXPECT_EQ(layouts.B_layout_, nullptr);
    EXPECT_EQ(layouts.C_layout_, nullptr);
    EXPECT_EQ(layouts.D_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, SetDimensionsWorks)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {};

    layouts.set_dimensions(1, 2, 3);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(LtLayoutsTests, CreateABDLayoutsCreates)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {};

    layouts.set_dimensions(8, 4, 2);

    EXPECT_TRUE(layouts.create_ABD_layouts<float>());

    EXPECT_NE(layouts.A_layout_, nullptr);
    EXPECT_NE(layouts.B_layout_, nullptr);
    EXPECT_NE(layouts.D_layout_, nullptr);
    EXPECT_EQ(layouts.C_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(LtLayoutsTests, CreateCLayoutCreates)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_NE(layouts.C_layout_, nullptr);
    EXPECT_EQ(layouts.A_layout_, nullptr);
    EXPECT_EQ(layouts.B_layout_, nullptr);
    EXPECT_EQ(layouts.D_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(LtLayoutsTests, SetBatchCountAndStridedOffsetsWorks)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_TRUE(layouts.set_batch_count_and_strided_offsets(
      2, 16, 8, 4));
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, GetBatchCountWorks)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_TRUE(layouts.set_batch_count_and_strided_offsets(
      2, 16, 8, 4));

    // Batch count is 2, size written is 4.
    EXPECT_EQ(layouts.get_batch_count('A'), std::make_tuple(2, 4));
    EXPECT_EQ(layouts.get_batch_count('B'), std::make_tuple(2, 4));
    EXPECT_EQ(layouts.get_batch_count('C'), std::make_tuple(2, 4));
    EXPECT_EQ(layouts.get_batch_count('D'), std::make_tuple(2, 4));
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, GetStridedBatchOffsetWorks)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_TRUE(layouts.set_batch_count_and_strided_offsets(
      2, 16, 8, 4));

    // Strided batch offset is 16, size written is 8.
    EXPECT_EQ(layouts.get_strided_batch_offset('A'), std::make_tuple(16, 8));
    EXPECT_EQ(layouts.get_strided_batch_offset('B'), std::make_tuple(8, 8));
    EXPECT_EQ(layouts.get_strided_batch_offset('C'), std::make_tuple(4, 8));
    EXPECT_EQ(layouts.get_strided_batch_offset('D'), std::make_tuple(4, 8));
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, GetMemoryOrderGets)
{
  LtLayouts layouts {8, 4, 2};

  EXPECT_TRUE(layouts.create_ABD_layouts<float>());

  EXPECT_EQ(
    layouts.get_memory_order('A'),
    std::make_tuple(CUBLASLT_ORDER_COL, 4));
  EXPECT_EQ(
    layouts.get_memory_order('B'),
    std::make_tuple(CUBLASLT_ORDER_COL, 4));
  EXPECT_EQ(
    layouts.get_memory_order('D'),
    std::make_tuple(CUBLASLT_ORDER_COL, 4));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, GetMemoryOrderErrorsOnUncreatedLayout)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());

    EXPECT_EQ(
      layouts.get_memory_order('C'),
      std::nullopt);
  }

  std::string expected_error_message {
    "Either sizeInBytes is 0 and sizeWritten is NULL, or \nsizeInBytes is "
    "non-zero and buf is NULL, or\nsizeInBytes doesn't match size of internal "
    "storage for the selected attribute.\n for attribute 1\n"};

  EXPECT_EQ(capture_cerr.local_oss_.str(), expected_error_message);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, SetMemoryOrderSets)
{
  LtLayouts layouts {8, 4, 2};
  EXPECT_TRUE(layouts.create_ABD_layouts<float>());

  EXPECT_TRUE(layouts.set_memory_order('A', CUBLASLT_ORDER_ROW));
  EXPECT_EQ(
    layouts.get_memory_order('A'),
    std::make_tuple(CUBLASLT_ORDER_ROW, 4));
  EXPECT_TRUE(layouts.set_memory_order('B', CUBLASLT_ORDER_ROW));
  EXPECT_EQ(
    layouts.get_memory_order('B'),
    std::make_tuple(CUBLASLT_ORDER_ROW, 4));

  EXPECT_TRUE(layouts.set_memory_order('D', CUBLASLT_ORDER_ROW));
  EXPECT_EQ(
    layouts.get_memory_order('D'),
    std::make_tuple(CUBLASLT_ORDER_ROW, 4));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtLayoutsTests, SetMemoryOrderErrorsOnUncreatedLayout)
{
  CaptureCerr capture_cerr {};

  {
    LtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());

    EXPECT_FALSE(layouts.set_memory_order('C', CUBLASLT_ORDER_ROW));
  }

  std::string expected_error_message {
    "buf is NULL or sizeInBytes doesn't match size of internal storage for the "
    "selected attribute for attribute 1\n"};

  EXPECT_EQ(capture_cerr.local_oss_.str(), expected_error_message);
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
