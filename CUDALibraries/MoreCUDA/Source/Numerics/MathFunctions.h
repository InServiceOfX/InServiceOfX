#ifndef NUMERICS_MATH_FUNCTIONS_H
#define NUMERICS_MATH_FUNCTIONS_H

#include <cuda_bf16.h>  // For bfloat16 support
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

//-----------------------------------------------------------------------------
/// It's very important to use the inline keyword for specializations to
/// linking errors due to multiple definitions.
//-----------------------------------------------------------------------------

template<> __device__ inline float get_exponential<float>(const float value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff
  return expf(value);
}

template<> __device__ inline double get_exponential<double>(const double value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43expd
  return exp(value);
}

// __half and __half2 require sm_53+ for device code (first arch with native
// half-precision math). Available in host code unconditionally.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

template<> __device__ inline __half get_exponential<__half>(const __half value)
{
  // hexp: declared in /usr/local/cuda/include/cuda_fp16.h line 4070:
  //   __CUDA_FP16_DECL__ __half hexp(const __half a)
  // See https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hexpK6__half
  return hexp(value);
}

//------------------------------------------------------------------------------
/// h2exp computes exp on both 16-bit lanes simultaneously in one instruction.
///
/// Declaration from /usr/local/cuda/include/cuda_fp16.h line 4362:
///   __CUDA_FP16_DECL__ __half2 h2exp(const __half2 a)
///
/// NVIDIA CUDA Math API documentation, verbatim from:
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__FUNCTIONS.html
///
///   Brief:   "Calculates half2 vector exponential function in
///             round-to-nearest-even mode."
///   Details: "Calculates half2 exponential function of input vector a in
///             round-to-nearest-even mode."
///   Returns: "half2 - The elementwise exponential function on vector a."
//------------------------------------------------------------------------------
template<> __device__ inline __half2 get_exponential<__half2>(const __half2 value)
{
  return h2exp(value);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

// __nv_bfloat16 native math requires sm_80+ (Ampere). bfloat16 is 1 sign +
// 8 exponent + 7 mantissa bits: float's full exponent range at ~2^-8
// relative precision (vs __half's 5 exponent + 10 mantissa).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template<> __device__ inline __nv_bfloat16 get_exponential<__nv_bfloat16>(
  const __nv_bfloat16 value)
{
  // hexp (bfloat16 overload): declared in /usr/local/cuda/include/cuda_bf16.h
  // line 3910:
  //   __CUDA_BF16_DECL__ __nv_bfloat16 hexp(const __nv_bfloat16 a)
  // See https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html
  return hexp(value);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

//------------------------------------------------------------------------------
/// Calculate the fast approximate base e exponential of the input argument x,
/// e^x. Returns an approximation to e^x.
///
/// float specialization uses __expf (CUDA single-precision intrinsic).
///
/// Declaration from /usr/local/cuda/include/crt/device_functions.h line 371:
///   __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__
///   float __expf(float x)
///
/// NVIDIA CUDA Math API documentation, verbatim from:
/// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
///
///   Brief:   "Calculate the fast approximate base e exponential of the
///             input argument."
///   Details: "Calculate the fast approximate base e exponential of the
///             input argument x, e^x."
///   Returns: "Returns an approximation to e^x."
///
/// __expf compiles to the ex2.approx hardware instruction (~4× faster than
/// expf, max error ≤ 2 ULP). Use in attention softmax kernels where
/// sub-ULP IEEE accuracy is not required (e.g. FlashAttention score scaling).
///
/// double: CUDA has no __exp double intrinsic. exp() already maps to the
/// hardware DFMA unit on sm_20+, so there is no approximate alternative.
///
/// __half/__half2: hexp()/h2exp() are the native half-precision exponential
/// instructions on sm_53+. h2exp computes both lanes simultaneously.
//------------------------------------------------------------------------------
template <typename FPType>
__device__ FPType get_approximate_exponential(const FPType value) = delete;

template<> __device__ inline float get_approximate_exponential<float>(const float value)
{
  // Declared in /usr/local/cuda/include/crt/device_functions.h line 371.
  return __expf(value);
}

template<> __device__ inline double get_approximate_exponential<double>(const double value)
{
  // No __exp double intrinsic exists in CUDA; fall back to full-precision exp().
  return exp(value);
}

// __half and __half2 require sm_53+ for device code (first arch with native
// half-precision math). Available in host code unconditionally.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

template<> __device__ inline __half get_approximate_exponential<__half>(const __half value)
{
  // hexp: declared in /usr/local/cuda/include/cuda_fp16.h line 4070:
  //   __CUDA_FP16_DECL__ __half hexp(const __half a)
  return hexp(value);
}

template<> __device__ inline __half2 get_approximate_exponential<__half2>(const __half2 value)
{
  // h2exp: declared in /usr/local/cuda/include/cuda_fp16.h line 4362:
  //   __CUDA_FP16_DECL__ __half2 h2exp(const __half2 a)
  // Processes both 16-bit lanes simultaneously (same instruction as
  // get_exponential<__half2> — no approximate vs exact distinction at half).
  return h2exp(value);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template<> __device__ inline __nv_bfloat16 get_approximate_exponential<
  __nv_bfloat16>(const __nv_bfloat16 value)
{
  // Same hexp bfloat16 overload as get_exponential<__nv_bfloat16>
  // (cuda_bf16.h line 3910) — at 7 mantissa bits there is no approximate vs
  // exact distinction.
  return hexp(value);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template <typename FPType>
__device__ FPType get_max(const FPType a, const FPType b) = delete;

template<> __device__ inline float get_max<float>(const float a, const float b)
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
template<> __device__ inline double get_max<double>(const double a, const double b)
{
  return fmax(a, b);
}

//------------------------------------------------------------------------------
/// Calculate ln(x), the base e (natural) logarithm of the input argument x.
/// Used to form the logsumexp statistic L = m + ln(ℓ) that the FlashAttention
/// backward pass stores per row for tile-wise recomputation of the attention
/// weights, P_ij = exp(S_ij − L_i).
//------------------------------------------------------------------------------
template <typename FPType>
__device__ FPType get_natural_log(const FPType value) = delete;

template<> __device__ inline float get_natural_log<float>(const float value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff
  return logf(value);
}

template<> __device__ inline double get_natural_log<double>(const double value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43logd
  return log(value);
}

template <typename FPType>
__device__ FPType get_sqrt(const FPType value) = delete;

template<> __device__ inline float get_sqrt<float>(const float value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html
  return sqrtf(value);
}

template<> __device__ inline double get_sqrt<double>(const double value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html
  return sqrt(value);
}

template<> __device__ inline __half get_sqrt<__half>(const __half value)
{
  // See
  // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html
  return hsqrt(value);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template<> __device__ inline __nv_bfloat16 get_sqrt<__nv_bfloat16>(
  const __nv_bfloat16 value)
{
  // hsqrt (bfloat16 overload): declared in /usr/local/cuda/include/
  // cuda_bf16.h line 3813:
  //   __CUDA_BF16_DECL__ __nv_bfloat16 hsqrt(const __nv_bfloat16 a)
  return hsqrt(value);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

} // namespace MathFunctions
} // namespace Numerics

#endif // NUMERICS_MATH_FUNCTIONS_H