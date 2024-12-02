#ifndef CUBLAS_WRAPPERS_GET_DATA_PRECISION_H
#define CUBLAS_WRAPPERS_GET_DATA_PRECISION_H

#include <cublasLt.h>

namespace cuBLASWrappers
{


//------------------------------------------------------------------------------
/// This function is meant to replace the use of macros.
/// https://github.com/karpathy/llm.c/blob/master/llmc/cublas_common.h#L14
/// 
//------------------------------------------------------------------------------
template <typename T>
constexpr cudaDataType_t get_data_precision() = delete;

template <>
inline constexpr cudaDataType_t get_data_precision<float>()
{
  return CUDA_R_32F;
}

template <>
inline constexpr cudaDataType_t get_data_precision<__half>()
{
  return CUDA_R_16F;
}

template <>
inline constexpr cudaDataType_t get_data_precision<__nv_bfloat16>()
{
  return CUDA_R_16BF;
}

} // namespace cuBLASWrappers
#endif // CUBLAS_WRAPPERS_GET_DATA_PRECISION_H
