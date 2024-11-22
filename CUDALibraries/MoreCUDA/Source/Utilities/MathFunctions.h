#ifndef UTILITIES_MATH_FUNCTIONS_H
#define UTILITIES_MATH_FUNCTIONS_H

#include <cuda_fp16.h>  // For half precision support

namespace Utilities
{
namespace MathFunctions
{

template <typename FPType>
__device__ FPType get_sqrt(const FPType value) = delete;

template<> __device__ float get_sqrt<float>(const float value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html
  return sqrtf(value);
}

template<> __device__ double get_sqrt<double>(const double value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html
  return sqrt(value);
}


template<> __device__ __half get_sqrt<__half>(const __half value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html
  return hsqrt(value);
}

} // namespace MathFunctions
} // namespace Utilities

#endif // UTILITIES_MATH_FUNCTIONS_H