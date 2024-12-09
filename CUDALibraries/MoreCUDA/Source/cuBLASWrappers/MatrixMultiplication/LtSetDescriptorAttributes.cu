#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"

#include <cstdint>
#include <cublasLt.h>
#include <iostream>

using std::cerr;

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

bool LtSetDescriptorAttributes::handle_set_descriptor_status(
  const cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      static constexpr const char* error_message_1 {
        "buf is NULL or sizeInBytes doesn't match the size of the internal "};
      static constexpr const char* error_message_2 {
        "storage for the selected attribute."};

      cerr << error_message_1 << error_message_2 << '\n';
      return false;
    }

    cerr << "Failed to set transpose on A\n";
    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescgetattribute
/// 3.4.28 cublasLtMatmulDescGetAttribute(..)
//------------------------------------------------------------------------------
bool LtSetDescriptorAttributes::handle_get_attribute(
const cublasStatus_t status)
{
  // attribute's value was successfully written to user memory.
  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else if (status == CUBLAS_STATUS_INVALID_VALUE)
  {
    cerr << "Either sizeInBytes is 0 and sizeWritten is NULL, or\n";
    cerr << "sizeInBytes is non-zero and buf is NULL, or\n";
    cerr << "sizeInBytes doesn't match size of internal storage for the ";
    cerr << " selected attribute.\n";
    return false;
  }
  else
  {
    cerr << "Failed to get attribute.\n";
    return false;
  }
}

bool LtSetDescriptorAttributes::set_transpose_on_A(
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

bool LtSetDescriptorAttributes::set_transpose_on_B(
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

std::optional<std::pair<int32_t, uint64_t>>
  LtSetDescriptorAttributes::get_transpose_operation_on_A(
    cublasLtMatmulDesc_t matmul_descriptor)
{
  int32_t transpose_operation;
  uint64_t size_in_bytes;

  const cublasStatus_t status {
    cublasLtMatmulDescGetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_TRANSA,
      &transpose_operation,
      sizeof(transpose_operation),
      &size_in_bytes)};

  if (!handle_get_attribute(status))
  {
    return std::nullopt;
  }

  return std::make_pair(transpose_operation, size_in_bytes);
}

std::optional<std::pair<int32_t, uint64_t>>
  LtSetDescriptorAttributes::get_transpose_operation_on_B(
    cublasLtMatmulDesc_t matmul_descriptor)
{
  int32_t transpose_operation;
  uint64_t size_in_bytes;

  const cublasStatus_t status {
    cublasLtMatmulDescGetAttribute(
      matmul_descriptor,
      CUBLASLT_MATMUL_DESC_TRANSB,
      &transpose_operation,
      sizeof(transpose_operation),
      &size_in_bytes)};

  if (!handle_get_attribute(status))
  {
    return std::nullopt;
  }

  return std::make_pair(transpose_operation, size_in_bytes);
}

bool LtSetDescriptorAttributes::set_gelu_epilogue_auxiliary_leading_dimension(
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

void LtSetDescriptorAttributes::set_epilogue(
  const bool has_gelu,
  const bool is_backward,
  const bool has_bias)
{
  epilogue_ = get_epilogue_postprocessing_options(
    has_gelu,
    is_backward,
    has_bias);
}

bool LtSetDescriptorAttributes::set_epilogue_function(
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

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t
/// 3.3.9. cublasLtMatmulDescAttributes_t
/// CUBLASLT_MATMUL_DESC_BIAS_POINTER entry.
//------------------------------------------------------------------------------

uint64_t LtSetDescriptorAttributes::get_bias_size(
  const uint64_t m,
  const uint64_t n,
  const bool has_gelu,
  const bool is_backward)
{
  if (has_gelu)
  {
    if (is_backward)
    {
      // We shouldn't have any backward matrix multiplication that uses both
      // GELU and bias.
      return 0;
    }
    else
    {
      // CUBLASLT_EPILOGUE_GELU_AUX_BIAS - Input vector with length matching
      // number of rows of matrix D.
      return m;
    }
  }
  else
  {
    // CUBLASLT_EPILOGUE_BGRADB - Output vector with length matching number of
    // columns of matrix D.
    // CUBLASLT_EPILOGUE_BIAS - Input vector with length matching number of rows
    // of matrix D.
    return is_backward ? n : m;
  }
}

bool LtSetDescriptorAttributes::set_scale_type(
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

cublasLtEpilogue_t LtSetDescriptorAttributes::get_epilogue_postprocessing_options(
  const bool has_gelu,
  const bool is_backward,
  const bool has_bias)
{
  if (has_gelu)
  {
    if (is_backward)
    {
      // We shouldn't have any backward matrix multiplications that use both
      // GELU and bias.
      if (!has_bias)
      {
        // CUBLASLT_EPILOGUE_DGELU = 64 | 128 - Apply GELU gradient to matmul
        // output. Store GELU gradient in output matrix. This epilogue mode
        // requires an extra input, see
        // CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER of
        // cublasLtMatmulDescAttributes_t.
        return CUBLASLT_EPILOGUE_DGELU;
      }
      else
      {
        // https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t
        // = CUBLASLT_EPILOGUE_DGELU | 16 - Apply independently GELU and Bias
        // gradient to matmul output. Store GELU gradient in output matrix, and
        // Bias gradient in bias buffer (see CUBLASLT_MATMUL_DEC_BIAS_POINTER).
        return CUBLASLT_EPILOGUE_DGELU_BGRAD;
      }
    }
    else
    {
      // CUBLASLT_EPILOGUE_GELU_AUX_BIAS =
      //   CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS
      // Apply Bias and then GELU transform. This epilogue mode outputs GELU
      // input as a separate matrix (useful for training).
      return has_bias ?
        CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
    }
  }
  else if (has_bias)
  {
    // https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t
    // CUBLASLT_EPILOGUE_BGRADB = 512 - Apply bias gradient to the input matrix
    // B. Bias size corresponds to number of columns of matrix D. Reduction
    // happens over the GEMM's "k" dimension. Store Bias gradient in the bias
    // buffer, see CUBLASLT_MATMUL_DESC_BIAS_POINTER.
    return is_backward ?
      CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
  }
  else
  {
    return CUBLASLT_EPILOGUE_DEFAULT;
  }
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
