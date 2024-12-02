#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtSetDescriptorAttributes.h"

#include <cstdint>
#include <cublasLt.h>
#include <iostream>
#include <string>

using std::cerr;
using std::string;

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescsetattribute
/// 3.4.29 cublasLtMatmulDescSetAttribute()
/// cublasStatus_t cublasLtMatmulDescSetAttribute(
///   cublasLtMatmulDesc_t matmulDesc,
///   cublasLtMatmulDescAttribute_t attr,
///   const void *buf,
///   size_t sizeInBytes)
/// It returns either CUBLAS_STATUS_SUCCESS or CUBLAS_STATUS_INVALID_VALUE from
/// documentation.
//------------------------------------------------------------------------------

bool cuBLASLtSetDescriptorAttributes::handle_set_descriptor_status(
  const cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      string error_message {
        "buf is NULL or sizeInBytes doesn't match the size of the internal "
        "storage for the selected attribute."};
      cerr << error_message << '\n';
      return false;
    }

    cerr << "Failed to set transpose on A\n";
    return false;
  }

  return true;
}

bool cuBLASLtSetDescriptorAttributes::set_transpose_on_A(
  cublasLtMatmulDesc_t matmul_descriptor,
  const bool is_transpose)
{
  // https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
  // CUBLASLT_MATMUL_DESC_TRANSA - Specifies type of transformation operation
  // that should be performed on matrix A. Default value is: CUBLAS_OP_N (i.e.,
  // non-transpose operation).
  // Data Type int32_t
  const cublasStatus_t status {
    cublasLtMatmulDescSetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_TRANSA,
      (is_transpose) ? &transpose_ : &no_transpose_,
      sizeof(transpose_))};

  return handle_set_descriptor_status(status);
}

bool cuBLASLtSetDescriptorAttributes::set_transpose_on_B(
  cublasLtMatmulDesc_t matmul_descriptor,
  const bool is_transpose)
{
  // https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
  // CUBLASLT_MATMUL_DESC_TRANSB - Specifies type of transformation operation
  // that should be performed on matrix B. Default value is: CUBLAS_OP_N (i.e.,
  // non-transpose operation).
  // Data Type int32_t
  const cublasStatus_t status {
    cublasLtMatmulDescSetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_TRANSB,
      (is_transpose) ? &transpose_ : &no_transpose_,
      sizeof(transpose_))};

  return handle_set_descriptor_status(status);
}

bool cuBLASLtSetDescriptorAttributes::set_gelu_epilogue_auxiliary_leading_dimension(
  cublasLtMatmulDesc_t matmul_descriptor,
  const int64_t m)
{
  if (m % 8 != 0)
  {
    cerr << "GELU input matrix leading dimension must be divisible by 8\n";
    return false;
  }

  gelu_leading_dimension_ = m;

  const cublasStatus_t status {
    cublasLtMatmulDescSetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
      &gelu_leading_dimension_,
      sizeof(gelu_leading_dimension_))};

  return handle_set_descriptor_status(status);
}

void cuBLASLtSetDescriptorAttributes::set_epilogue(
  const bool has_gelu,
  const bool is_backward,
  const bool has_bias)
{
  if (has_gelu)
  {
    if (is_backward)
    {
      // We shouln't have any backward matrix multiplications that use both
      // GELU and bias.
      if (!has_bias)
      {
        // CUBLASLT_EPILOGUE_DGELU = 64 | 128 - Apply GELU gradient to matmul
        // output. Store GELU gradient in output matrix. This epilogue mode
        // requires an extra input, see
        // CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of
        // cublasLtMatmulDescAttributes_t.
        epilogue_ = CUBLASLT_EPILOGUE_DGELU;
      }
    }
    else
    {
      // CUBLASLT_EPILOGUE_GELU_AUX_BIAS =
      //   CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS
      // Apply Bias and then GELU transform. This epilogue mode outputs GELU
      // input as a separate matrix (useful for training).
      epilogue_ = has_bias ?
        CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
    }
  }
  else if (has_bias)
  {
    epilogue_ = is_backward ?
      CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
  }
  else
  {
    epilogue_ = CUBLASLT_EPILOGUE_DEFAULT;
  }
}

bool cuBLASLtSetDescriptorAttributes::set_epilogue_function(
  cublasLtMatmulDesc_t matmul_descriptor)
{
  const cublasStatus_t status {
    cublasLtMatmulDescSetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue_,
      sizeof(epilogue_))};

  return handle_set_descriptor_status(status);
}

bool cuBLASLtSetDescriptorAttributes::set_scale_type(
  cublasLtMatmulDesc_t matmul_descriptor,
  const cublasDataType_t scale_type)
{
  scale_type_ = scale_type;

  return handle_set_descriptor_status(
    cublasLtMatmulDescSetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_SCALE_TYPE,
      &scale_type_,
      sizeof(scale_type_)));
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
