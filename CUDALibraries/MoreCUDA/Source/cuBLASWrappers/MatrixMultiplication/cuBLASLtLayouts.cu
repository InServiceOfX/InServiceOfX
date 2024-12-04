#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtLayouts.h"
#include "cuBLASWrappers/get_data_precision.h"

#include <cstdint>
#include <cublasLt.h>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>

using std::cerr;

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatrixlayoutcreate
/// cublasStatus_t cublasLtMatrixLayoutCreate(
///   cublasLtMatrixLayout_t *matLayout,
///   cublasDataType_t type,
///   uint64_t rows,
///   uint64_t cols,
///   int64_t ld)
/// where
/// rows, cols - Input - Number of rows and columns of matrix.
/// ld - Input - Leading dimension of matrix. In column major layout, this is
/// the number of elements to jump to reach the next column. Thus ld >= m
/// (number of rows)
//------------------------------------------------------------------------------

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

cuBLASLtLayouts::~cuBLASLtLayouts()
{
  destroy_layouts();
}

void cuBLASLtLayouts::set_dimensions(
  const uint64_t m,
  const uint64_t n,
  const uint64_t k)
{
  m_ = m;
  n_ = n;
  k_ = k;
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutsetattribute
/// 3.4.39. cublasLtMatrixLayoutSetAttribute
/// cublasStatus_t cublasLtMatrixLayoutSetAttribute(
///   cublasLtMatrixLayout_t matLayout,
///   cublasLtMatrixLayoutAttribute_t attr,
///   const void *buf,
///   size_t sizeInBytes)
/// where
/// matLayout - Input - Pointer to previously created structure holding the
/// matrix layout descriptor queried by this function.
/// attr - Input - Attribute that'll be set by this function.
/// See cublasLtMatrixLayoutAttribute_t.
/// buf - Input - Value to which specified attribute should be set.
/// sizeInBytes - Input - Size of buf, the attribute buffer.
///
/// https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutattribute-t
/// CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT Number of matmul operations to perform
/// in batch. Default value is 1. int32_t
/// CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET Stride (in elements) to next
/// matrix for strided batch operation. Default value is 0. int64_t
//------------------------------------------------------------------------------

bool cuBLASLtLayouts::set_batch_count_and_strided_offsets(
  const int32_t batch_count,
  const int64_t A_strided_batch_offset,
  const int64_t B_strided_batch_offset,
  const int64_t output_strided_batch_offset)
{
  batch_count_ = batch_count;

  const bool status_A_batch_count {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        A_layout_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count_,
        sizeof(batch_count_)),
      CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)};

  if (!status_A_batch_count)
  {
    return false;
  }

  const bool status_B_batch_count {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        B_layout_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count_,
        sizeof(batch_count_)),
      CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)};

  if (!status_B_batch_count)
  {
    return false;
  }

  const bool status_C_batch_count {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        C_layout_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count_,
        sizeof(batch_count_)),
      CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)};

  if (!status_C_batch_count)
  {
    return false;
  }

  const bool status_D_batch_count {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        D_layout_,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count_,
        sizeof(batch_count_)),
      CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)};

  if (!status_D_batch_count)
  {
    return false;
  }

  A_strided_batch_offset_ = A_strided_batch_offset;

  const bool status_A_strided_batch_offset {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        A_layout_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &A_strided_batch_offset_,
        sizeof(A_strided_batch_offset_)),
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)};

  if (!status_A_strided_batch_offset)
  {
    return false;
  }

  B_strided_batch_offset_ = B_strided_batch_offset;

  const bool status_B_strided_batch_offset {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        B_layout_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &B_strided_batch_offset_,
        sizeof(B_strided_batch_offset_)),
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)};

  if (!status_B_strided_batch_offset)
  {
    return false;
  }

  output_strided_batch_offset_ = output_strided_batch_offset;

  const bool status_C_strided_batch_offset {
    handle_set_attribute(
      cublasLtMatrixLayoutSetAttribute(
        C_layout_,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &output_strided_batch_offset_,
        sizeof(output_strided_batch_offset_)),
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)};

  if (!status_C_strided_batch_offset)
  {
    return false;
  }

  return handle_set_attribute(
    cublasLtMatrixLayoutSetAttribute(
      D_layout_,
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
      &output_strided_batch_offset_,
      sizeof(output_strided_batch_offset_)),
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
}

bool cuBLASLtLayouts::destroy_layouts()
{
  cublasStatus_t status_A {cublasLtMatrixLayoutDestroy(A_layout_)};
  cublasStatus_t status_B {cublasLtMatrixLayoutDestroy(B_layout_)};
  cublasStatus_t status_C {cublasLtMatrixLayoutDestroy(C_layout_)};
  cublasStatus_t status_D {cublasLtMatrixLayoutDestroy(D_layout_)};

  if (status_A != CUBLAS_STATUS_SUCCESS)
  {
    cerr << "cublasLtMatrixLayoutDestroy(A_layout_) failed.\n";
    return false;
  }
  else if (status_B != CUBLAS_STATUS_SUCCESS)
  {
    cerr << "cublasLtMatrixLayoutDestroy(B_layout_) failed.\n";
    return false;
  }
  else if (status_C != CUBLAS_STATUS_SUCCESS)
  {
    cerr << "cublasLtMatrixLayoutDestroy(C_layout_) failed.\n";
    return false;
  }
  else if (status_D != CUBLAS_STATUS_SUCCESS)
  {
    cerr << "cublasLtMatrixLayoutDestroy(D_layout_) failed.\n";
    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatrixlayoutgetattribute
/// 3.4.38. cublasLtMatrixLayoutGetAttribute()
/// cublasStatus_t cublasLtMatrixLayoutGetAttribute(
///   cublasLtMatrixLayout_t matLayout,
///   cublasLtMatrixLayoutAttribute_t attr,
///   void *buf,
///   size_t sizeInBytes,
///   size_t *sizeWritten)
/// where
/// sizeInBytes - Input - Size of buf (in bytes) for verification.
/// sizeWritten - Output - Valid only when the return value is
/// CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: then sizeWritten is the
/// number of bytes actually written;
//------------------------------------------------------------------------------
std::optional<std::tuple<int32_t, uint64_t>> cuBLASLtLayouts::get_batch_count(
  const char matrix_name) const
{
  int32_t batch_count {0};
  uint64_t size_written {0};

  const bool status {handle_get_attribute(
    cublasLtMatrixLayoutGetAttribute(
      matrix_name == 'A' ? A_layout_ : matrix_name == 'B' ? B_layout_ :
      matrix_name == 'C' ? C_layout_ : D_layout_,
      CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
      &batch_count,
      sizeof(batch_count),
      &size_written),
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)};

  if (status)
  {
    return std::make_tuple(batch_count, size_written);
  }
  else
  {
    return std::nullopt;
  }
}

std::optional<std::tuple<int64_t, uint64_t>> cuBLASLtLayouts::get_strided_batch_offset(
  const char matrix_name) const
{
  int64_t strided_batch_offset {0};
  uint64_t size_written {0};

  const bool status {handle_get_attribute(
    cublasLtMatrixLayoutGetAttribute(
      matrix_name == 'A' ? A_layout_ : matrix_name == 'B' ? B_layout_ :
      matrix_name == 'C' ? C_layout_ : D_layout_,
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
      &strided_batch_offset,
      sizeof(strided_batch_offset),
      &size_written),
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)};

  if (status)
  {
    return std::make_tuple(strided_batch_offset, size_written);
  }
  else
  {
    return std::nullopt;
  }
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatrixlayoutcreate
//------------------------------------------------------------------------------
bool cuBLASLtLayouts::handle_create_layout_status(
  const cublasStatus_t status,
  const char matrix_name)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_ALLOC_FAILED)
    {
      cerr << "Memory could not be allocated for " << matrix_name << ".\n";
    }
    else
    {
      cerr << "Failed to create " << matrix_name << " layout.\n";
    }

    return false;
  }

  return true;
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatrixlayoutcreate
//------------------------------------------------------------------------------
bool cuBLASLtLayouts::handle_set_attribute(
  const cublasStatus_t status,
  cublasLtMatrixLayoutAttribute_t attribute)
{
  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else
  {
    if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      static constexpr const char* error_message_1 {
        "buf is NULL or sizeInBytes doesn't match size of internal storage "};
      static constexpr const char* error_message_2 {
        "for the selected attribute"};

      cerr << error_message_1 << error_message_2 << " for attribute "
        << attribute << "\n";
    }
    else
    {
      cerr << "Failed to set " << attribute << " attribute.\n";
    }

    return false;
  }
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatrixlayoutgetattribute
//------------------------------------------------------------------------------
bool cuBLASLtLayouts::handle_get_attribute(
  const cublasStatus_t status,
  cublasLtMatrixLayoutAttribute_t attribute)
{
  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else
  {
    if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      static constexpr const char* error_message_1 {
        "Either sizeInBytes is 0 and sizeWritten is NULL, or \n"};
      static constexpr const char* error_message_2 {
        "sizeInBytes is non-zero and buf is NULL, or\n"};
      static constexpr const char* error_message_3 {
        "sizeInBytes doesn't match size of internal storage for the selected "};
      static constexpr const char* error_message_4 {
        "attribute.\n"};

      cerr << error_message_1 << error_message_2 << error_message_3
        << error_message_4 << " for attribute " << attribute << "\n";
    }
    else
    {
      cerr << "Failed to set " << attribute << " attribute.\n";
    }

    return false;
  }
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
