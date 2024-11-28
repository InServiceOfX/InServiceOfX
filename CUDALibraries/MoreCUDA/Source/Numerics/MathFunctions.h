#ifndef NUMERICS_MATH_FUNCTIONS_H
#define NUMERICS_MATH_FUNCTIONS_H

#include <cuda_fp16.h>  // For half precision support

namespace Numerics
{
namespace MathFunctions
{

//------------------------------------------------------------------------------
/// Calculate e^x, the base e exponential of the input argument x.
//------------------------------------------------------------------------------
template <typename FPType>
__device__ FPType get_exponential(const FPType value) = delete;

template<> __device__ float get_exponential<float>(const float value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff
  return expf(value);
}

template<> __device__ double get_exponential<double>(const double value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43expd
  return exp(value);
}

// TODO: Make a macro or decide to stop supporting compute capability 5.X or
// earlier.
#ifndef __CUDA_ARCH__

template<> __device__ __half get_exponential<__half>(const __half value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hexpK6__half
  return hexp(value);
}

#endif // __CUDA_ARCH__

template <typename FPType>
__device__ FPType get_max(const FPType a, const FPType b) = delete;

template<> __device__ float get_max<float>(const float a, const float b)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmaxfff
  return fmaxf(a, b);
}

//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmaxdd
/// Treats NaN arguments as missing data. If 1 argument is NaN and the other is
/// legitamate numeric value, numeric value is chosen.
//------------------------------------------------------------------------------
template<> __device__ double get_max<double>(const double a, const double b)
{
  return fmax(a, b);
}

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
} // namespace Numerics

#endif // NUMERICS_MATH_FUNCTIONS_H