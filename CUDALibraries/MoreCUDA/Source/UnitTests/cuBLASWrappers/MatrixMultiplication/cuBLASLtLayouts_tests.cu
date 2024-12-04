#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtLayouts.h"
#include "Utilities/CaptureCerr.h"

#include "gtest/gtest.h"

#include <tuple>

using cuBLASWrappers::MatrixMultiplication::cuBLASLtLayouts;
using Utilities::CaptureCerr;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

TEST(cuBLASLtLayoutsTests, DefaultConstructor)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {};

    EXPECT_EQ(layouts.A_layout_, nullptr);
    EXPECT_EQ(layouts.B_layout_, nullptr);
    EXPECT_EQ(layouts.C_layout_, nullptr);
    EXPECT_EQ(layouts.D_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(cuBLASLtLayoutsTests, SetDimensionsWorks)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {};

    layouts.set_dimensions(1, 2, 3);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(cuBLASLtLayoutsTests, CreateABDLayoutsCreates)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {};

    layouts.set_dimensions(8, 4, 2);

    EXPECT_TRUE(layouts.create_ABD_layouts<float>());

    EXPECT_NE(layouts.A_layout_, nullptr);
    EXPECT_NE(layouts.B_layout_, nullptr);
    EXPECT_NE(layouts.D_layout_, nullptr);
    EXPECT_EQ(layouts.C_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(cuBLASLtLayoutsTests, CreateCLayoutCreates)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_NE(layouts.C_layout_, nullptr);
    EXPECT_EQ(layouts.A_layout_, nullptr);
    EXPECT_EQ(layouts.B_layout_, nullptr);
    EXPECT_EQ(layouts.D_layout_, nullptr);
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(cuBLASLtLayoutsTests, SetBatchCountAndStridedOffsetsWorks)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {8, 4, 2};
    EXPECT_TRUE(layouts.create_ABD_layouts<float>());
    EXPECT_TRUE(layouts.create_C_layout<float>());

    EXPECT_TRUE(layouts.set_batch_count_and_strided_offsets(
      2, 16, 8, 4));
  }

  EXPECT_EQ(capture_cerr.local_oss_.str(), "");
}

TEST(cuBLASLtLayoutsTests, GetBatchCountWorks)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {8, 4, 2};
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

TEST(cuBLASLtLayoutsTests, GetStridedBatchOffsetWorks)
{
  CaptureCerr capture_cerr {};

  {
    cuBLASLtLayouts layouts {8, 4, 2};
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

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
