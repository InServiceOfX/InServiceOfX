#ifndef UTILITIES_NUMERICAL_CONSTANTS_GET_INFINITY_H
#define UTILITIES_NUMERICAL_CONSTANTS_GET_INFINITY_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half precision support

namespace Utilities
{
namespace NumericalConstants
{

// Primary template - deleted to prevent instantiation with unsupported types.
template<typename T> __device__ T get_infinity() = delete;

//------------------------------------------------------------------------------
/// Specialization for float
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html?highlight=__int_as_float#_CPPv414__int_as_floati
/// __device__ float __int_as_float(int x)
/// Reinterpret bits in an integer as a float.
//------------------------------------------------------------------------------
template<> __device__ float get_infinity<float>() 
{
  // 0x7f800000 represents infinity in IEEE-754 single precision
  // See also
  // https://github.com/tpn/cuda-samples/blob/master/v12.0/include/math_constants.h#L54
  return __int_as_float(0x7f800000U);
}

//------------------------------------------------------------------------------
/// Specialization for double
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html?highlight=__int_as_float#_CPPv420__longlong_as_doublex
/// __device__ double __longlong_as_double(long long int x)
/// Reinterpret bits in a 64-bit signed integer to a double.
//------------------------------------------------------------------------------
template<> __device__ double get_infinity<double>() 
{
  // 0x7ff0000000000000LL represents infinity in IEEE-754 double precision
  // See also
  // https://github.com/tpn/cuda-samples/blob/master/v12.0/include/math_constants.h#L91
  return __longlong_as_double(0x7ff0000000000000ULL);
}

//------------------------------------------------------------------------------
/// Specialization for half
/// See
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html
/// for type __half.
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF__CONSTANTS.html
/// Use the macro instead,
/// CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
//------------------------------------------------------------------------------
template<> __device__ __half get_infinity<__half>() 
{
  return __ushort_as_half((unsigned short)0x7C00U);
}

} // namespace NumericalConstants
} // namespace Utilities

#endif // UTILITIES_NUMERICAL_CONSTANTS_GET_INFINITY_H
