//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api
/// cuBLASLt library is a new lightweight library dedicated to GEneral
/// Matrix-to-matrix Multiply (GEMM) operations with a new, flexible API.
//------------------------------------------------------------------------------

#ifndef OPERATIONS_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_WITH_CUBLASLT_H
#define OPERATIONS_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_WITH_CUBLASLT_H

#include "NVIDIAToolsExtension/NVTXRange.h"

#include <cstdint> // uintptr_t

namespace Operations
{
namespace MatrixMultiplication
{

template <typename FPType>
void matrix_multiplication_with_cuBLASLt(
    FPType* d_D,
    const FPType* d_A,
    const FPType* d_B,
    const FPType* bias,
    const int M,
    const int N,
    const int K,
    FPType* pre_gelu=nullptr,
    const bool is_backward=false)
{
  NVTX_RANGE_FN();

  const bool has_bias {bias != nullptr};
  const bool is_gelu {pre_gelu != nullptr}; 

  // Check alignment (some modes work unaligned but it's always best to be
  // aligned for performance)
  // https://en.cppreference.com/w/cpp/types/integer
  // uintptr_t - unsigned integer type capable of holding a pointer to void.
  if ((
    (static_cast<uintptr_t>(d_A) % 16) != 0 ||
    (static_cast<uintptr_t>(d_B) % 16) != 0 ||
    (static_cast<uintptr_t>(d_D) % 16) != 0 ||
    (static_cast<uintptr_t>(bias) % 16) != 0))
  {
    throw std::runtime_error("All cuBLASlt pointers must be aligned!");
  }
    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc {};
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

}

} // namespace MatrixMultiplication
} // namespace Operations

#endif // OPERATIONS_MATRIX_MULTIPLICATION_MATRIX_MULTIPLICATION_WITH_CUBLASLT_H
