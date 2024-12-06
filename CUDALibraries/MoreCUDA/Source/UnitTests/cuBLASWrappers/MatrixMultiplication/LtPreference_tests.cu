#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "gtest/gtest.h"

using cuBLASWrappers::MatrixMultiplication::LtPreference;
using cuBLASWrappers::MatrixMultiplication::Workspace;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtPreferenceTests, DefaultConstructs)
{
  LtPreference preference {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtPreferenceTests, SetAttributeWithWorkspace)
{
  LtPreference preference {};
  Workspace workspace {};

  EXPECT_TRUE(preference.set_max_workspace_memory(
    workspace.workspace_size_in_bytes_));
}


} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests