#include "StreamManagement/Stream.h"

#include <gtest/gtest.h>

using StreamManagement::Stream;

namespace GoogleUnitTests
{

namespace StreamManagement
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Stream, Constructs)
{
  Stream stream;
  EXPECT_TRUE(stream.stream_ != nullptr);
}

} // namespace StreamManagement
} // namespace GoogleUnitTests
