#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "gtest/gtest.h"

using cuBLASWrappers::MatrixMultiplication::Workspace;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WorkspaceTests, DefaultConstructs)
{
  Workspace workspace {};

  SUCCEED();
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
