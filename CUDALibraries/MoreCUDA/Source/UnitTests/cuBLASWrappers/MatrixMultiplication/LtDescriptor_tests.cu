#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"

#include "gtest/gtest.h"

#include <cublasLt.h>

using cuBLASWrappers::MatrixMultiplication::ComputeParameters;
using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::get_compute_parameters;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtDescriptorComputeParametersTests, DefaultConstructs)
{
  ComputeParameters compute_parameters {};

  EXPECT_EQ(compute_parameters.compute_precision_mode_, CUBLAS_COMPUTE_32F);
  EXPECT_EQ(compute_parameters.data_type_, CUDA_R_32F);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtDescriptorComputeParametersTests, GetComputeParametersWorksForDouble)
{
  ComputeParameters compute_parameters {get_compute_parameters<double>()};

  EXPECT_EQ(compute_parameters.compute_precision_mode_, CUBLAS_COMPUTE_64F);
  EXPECT_EQ(compute_parameters.data_type_, CUDA_R_64F);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtDescriptorTests, DefaultConstructs)
{
  LtDescriptor descriptor {};

  EXPECT_NE(descriptor.descriptor_, nullptr);
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
