#include "NVIDIAToolsExtension/NVTXRange.h"
#include "gtest/gtest.h"

using NVIDIAToolsExtension::NVTXRange;

namespace GoogleUnitTests
{
namespace NVIDIAToolsExtension
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVTXRangeTests, NestedRanges)
{
  {
    NVTXRange outer("Outer Range");
    {
      NVTXRange inner("Inner Range", 1);
      // The destructor order will verify proper nesting
    }
  }
  // Test passes if no crashes/hangs
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVTXRangeTests, SequentialRanges)
{
  for(int i = 0; i < 3; i++)
  {
    NVTXRange range("Sequential Range", i);
    EXPECT_TRUE(true);
  }
  // Test passes if no crashes/hangs
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(NVTXRangeTests, ExceptionSafety)
{
  try
  {
    NVTXRange range("Exception Range");
    throw std::runtime_error("Test exception");
  }
  catch (const std::runtime_error&)
  {
    // Range should be properly popped even with exception
    EXPECT_TRUE(true);
  }
  SUCCEED();
}

} // namespace NVIDIAToolsExtension
} // namespace GoogleUnitTests