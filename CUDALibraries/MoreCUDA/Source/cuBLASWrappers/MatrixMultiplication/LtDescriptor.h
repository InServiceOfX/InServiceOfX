#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_DESCRIPTOR_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_DESCRIPTOR_H

#include <cublasLt.h>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

struct ComputeParameters
{
  // https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t
  // 2.2.11 cublasComputeType_t
  // cublasComputeType_t enumerate type used in cublasGemmEx() and
  // cublasLtMatmul() to choose compute precision modes.
  cublasComputeType_t compute_precision_mode_;
  // https://docs.nvidia.com/cuda/cublas/#cuda-datatypes-reference
  // 2.3.1 cudaDataType_t
  // cudaDataType_t type is an enumerant to specify data precision. It's used
  // when data reference doesn't carry type itself (e.g. void *)
  cudaDataType_t data_type_;

  ComputeParameters():
    // CUBLAS_COMPUTE_32F - This is default 32-bit single precision floating
    // point and uses compute and intermediate storage precisions of at least
    // 32-bits.
    compute_precision_mode_(CUBLAS_COMPUTE_32F),
    // CUDA_R_32F - 32-bit real single precision floating-point.
    data_type_(CUDA_R_32F)
  {}

  ComputeParameters(
    const cublasComputeType_t compute_precision_mode,
    const cudaDataType_t data_type):
    compute_precision_mode_{compute_precision_mode},
    data_type_{data_type}
  {}
};

template<typename T>
ComputeParameters get_compute_parameters() = delete;

template<>
inline ComputeParameters get_compute_parameters<float>()
{
  return ComputeParameters {};
}

template<>
inline ComputeParameters get_compute_parameters<double>()
{
  // CUBLAS_COMPUTE_64F - This is the default 64-bit double precision floating
  // point and uses compute and intermediate storage precisions of at least
  // 64-bits.
  // CUDA_R_64F - 64-bit real double precision floating-point.
  return ComputeParameters {CUBLAS_COMPUTE_64F, CUDA_R_64F};
}

template<>
inline ComputeParameters get_compute_parameters<__half>()
{
  // CUBLAS_COMPUTE_16F - This is the default and highest-performance mode for
  // 16-bit half precision floating point and all compute and intermediate
  // storage precisions with at least 16-bit half precision. Tensor Cores will
  // be used whenever possible.
  // CUDA_R_16F - 16-bit real half precision floating-point.
  return ComputeParameters {CUBLAS_COMPUTE_16F, CUDA_R_16F};
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldesc-t
/// cublasLtMatmulDesc_t type is a pointer to an opaque structure holding the
/// description of the matrix multiplication operation cublasLtMatmul().
//------------------------------------------------------------------------------
class LtDescriptor
{
  public:

    // Defaults to 32-bit single precision floating point data type.
    LtDescriptor();
    LtDescriptor(const ComputeParameters compute_parameters);
    ~LtDescriptor();

    cublasLtMatmulDesc_t descriptor_;

  protected:

    // \return true if successful, false otherwise.
    bool create_descriptor(const ComputeParameters compute_parameters);
    // \return true if successful, false otherwise.
    bool destroy_descriptor();
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_CUBLAS_LT_DESCRIPTOR_H  