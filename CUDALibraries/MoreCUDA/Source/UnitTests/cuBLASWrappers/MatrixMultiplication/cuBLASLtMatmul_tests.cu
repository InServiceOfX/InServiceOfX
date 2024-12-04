#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "gtest/gtest.h"

using cuBLASWrappers::MatrixMultiplication::cuBLASLtDescriptor;
using cuBLASWrappers::MatrixMultiplication::Workspace;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, Setup)
{
  cuBLASLtDescriptor descriptor {};
  Workspace workspace {};

}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
