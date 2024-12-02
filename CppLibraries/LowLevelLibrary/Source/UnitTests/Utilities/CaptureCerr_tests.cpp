#include "Utilities/CaptureCerr.h"

#include <gtest/gtest.h>
#include <iostream> // std::cerr
#include <sstream>
#include <string>

using Utilities::CaptureCerr;
using Utilities::capture_cerr;
using std::cerr;
using std::ostringstream;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Testing
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCerrTests, CaptureCerrAcceptsLocalOStringStream)
{
  ostringstream local_oss;

  auto cerr_buffer_ptr = capture_cerr(local_oss);

  cerr << "some message";

  cerr.rdbuf(cerr_buffer_ptr);

  const std::string expected_message {"some message"};
  EXPECT_EQ(local_oss.str(), expected_message);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCerrTests, DefaultConstructs)
{
  CaptureCerr capture_cerr {};
  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CaptureCerrTests, CaptureLocallyUponConstructionCapturesLocally)
{
  CaptureCerr capture_cerr {};
  cerr << "\n Testing Testing \n";

  EXPECT_EQ(capture_cerr.local_oss_.str(), "\n Testing Testing \n");
}

} // namespace Testing
} // namespace Utilities
} // namespace GoogleUnitTests
