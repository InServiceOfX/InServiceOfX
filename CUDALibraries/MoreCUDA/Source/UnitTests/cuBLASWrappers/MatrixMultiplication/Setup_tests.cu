#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "gtest/gtest.h"

// Use full name for Setup because there is a testing::Test::Setup

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupTests, DefaultConstructs)
{
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {};

  SUCCEED();
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests