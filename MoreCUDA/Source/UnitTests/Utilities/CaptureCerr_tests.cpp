#include "UnitTests/Utilities/CaptureCerr.h"

#include "gtest/gtest.h"
#include <iostream> // std::cerr

using std::cerr;
using UnitTests::Utilities::CaptureCerr;

namespace GoogleUnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCerrTests, DefaultConstructs)
{
	CaptureCerr capture_cerr {};

  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCerrTests, CaptureLocallyUponConstruction)
{
  CaptureCerr capture_cerr {};
  cerr << "\n Testing Testing \n";

  EXPECT_EQ(capture_cerr.local_oss_.str(), "\n Testing Testing \n");
}

} // namespace Utilities
} // namespace GoogleUnitTests