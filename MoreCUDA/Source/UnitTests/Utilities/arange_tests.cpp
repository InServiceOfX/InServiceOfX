#include "Utilities/arange.h"

#include "gtest/gtest.h"
#include <cstddef>

using Utilities::arange;

namespace GoogleUnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(arangeTests, CreatesRangeForStartStopAndStep)
{
  const auto result = arange<double>(0.0, 5.0, 0.5);

  EXPECT_EQ(result.size(), 10);

  EXPECT_DOUBLE_EQ(result.at(0), 0.0);
  EXPECT_DOUBLE_EQ(result.at(1), 0.5);
  EXPECT_DOUBLE_EQ(result.at(2), 1.0);
  EXPECT_DOUBLE_EQ(result.at(3), 1.5);
  EXPECT_DOUBLE_EQ(result.at(4), 2.0);
  EXPECT_DOUBLE_EQ(result.at(5), 2.5);
  EXPECT_DOUBLE_EQ(result.at(6), 3.0);
  EXPECT_DOUBLE_EQ(result.at(7), 3.5);
  EXPECT_DOUBLE_EQ(result.at(8), 4.0);
  EXPECT_DOUBLE_EQ(result.at(9), 4.5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(arangeTests, CreatesRangeForIntegersGivenSizeOnly)
{
  const auto result = arange<int>(3);

  EXPECT_EQ(result.size(), 3);

  EXPECT_EQ(result.at(0), 0);
  EXPECT_EQ(result.at(1), 1);
  EXPECT_EQ(result.at(2), 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(arangeTests, CreatesRangeForFloatGivenSizeOnly)
{
  const auto result = arange<float>(4);

  EXPECT_EQ(result.size(), 4);

  EXPECT_FLOAT_EQ(result.at(0), 0.0);
  EXPECT_FLOAT_EQ(result.at(1), 1.0);
  EXPECT_FLOAT_EQ(result.at(2), 2.0);
  EXPECT_FLOAT_EQ(result.at(3), 3.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(arangeTests, CreatesRangeForFloatGivenLvalueSize)
{
  const std::size_t N {9};

  const auto result = arange<float>(N);

  EXPECT_EQ(result.size(), 9);
  EXPECT_FLOAT_EQ(result.at(0), 0.0);
  EXPECT_FLOAT_EQ(result.at(1), 1.0);
  EXPECT_FLOAT_EQ(result.at(2), 2.0);
  EXPECT_FLOAT_EQ(result.at(3), 3.0);
  EXPECT_FLOAT_EQ(result.at(4), 4.0);
  EXPECT_FLOAT_EQ(result.at(5), 5.0);
  EXPECT_FLOAT_EQ(result.at(6), 6.0);
  EXPECT_FLOAT_EQ(result.at(7), 7.0);
  EXPECT_FLOAT_EQ(result.at(8), 8.0);
}

} // namespace Utilities
} // namespace GoogleUnitTests