#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtDescriptor.h"

#include "gtest/gtest.h"

#include <cublasLt.h>

using cuBLASWrappers::MatrixMultiplication::ComputeParameters;
using cuBLASWrappers::MatrixMultiplication::cuBLASLtDescriptor;
using cuBLASWrappers::MatrixMultiplication::get_compute_parameters;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtDescriptorComputeParametersTests, DefaultConstructs)
{
  ComputeParameters compute_parameters {};

  EXPECT_EQ(compute_parameters.compute_precision_mode_, CUBLAS_COMPUTE_32F);
  EXPECT_EQ(compute_parameters.data_type_, CUDA_R_32F);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  cuBLASLtDescriptorComputeParametersTests,
  GetComputeParametersWorksForDouble)
{
  ComputeParameters compute_parameters {get_compute_parameters<double>()};

  EXPECT_EQ(compute_parameters.compute_precision_mode_, CUBLAS_COMPUTE_64F);
  EXPECT_EQ(compute_parameters.data_type_, CUDA_R_64F);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtDescriptorTests, DefaultConstructs)
{
  cuBLASLtDescriptor descriptor {};

  EXPECT_NE(descriptor.descriptor_, nullptr);
}

} // namespace cuBLASWrappers
} // namespace GoogleUnitTests