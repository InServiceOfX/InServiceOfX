#include "cuBLASWrappers/LibraryContextHandle.h"

#include "gtest/gtest.h"

using cuBLASWrappers::LibraryContextHandle;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LibraryContextHandleTests, Constructs)
{
  LibraryContextHandle handle {};

  SUCCEED();
}

} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
