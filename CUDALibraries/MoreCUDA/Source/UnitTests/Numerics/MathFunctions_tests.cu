#include "DataStructures/Array.h"
#include "Numerics/MathFunctions.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <vector>

using DataStructures::Array;
using std::vector;
using Numerics::MathFunctions::get_approximate_exponential;
using Numerics::MathFunctions::get_exponential;
using Numerics::MathFunctions::get_max;
using Numerics::MathFunctions::get_sqrt;

namespace GoogleUnitTests
{
namespace Numerics
{
namespace MathFunctions
{

//------------------------------------------------------------------------------
// Kernels
//------------------------------------------------------------------------------

template <typename T>
__global__ void apply_get_exponential(T* output, const T* input, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    output[idx] = get_exponential<T>(input[idx]);
  }
}

template <typename T>
__global__ void apply_get_approximate_exponential(T* output, const T* input, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    output[idx] = get_approximate_exponential<T>(input[idx]);
  }
}

template <typename T>
__global__ void apply_get_max(T* output, const T* a, const T* b, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    output[idx] = get_max<T>(a[idx], b[idx]);
  }
}

template <typename T>
__global__ void apply_get_sqrt(T* output, const T* input, const int N)
{
  const int idx {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < N)
  {
    output[idx] = get_sqrt<T>(input[idx]);
  }
}

//------------------------------------------------------------------------------
// get_exponential tests
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetExponentialTests, FloatMatchesHostExpf)
{
  // GPU expf is correctly rounded (≤1 ULP); host expf is also correctly
  // rounded. EXPECT_FLOAT_EQ allows 4 ULP so both the exact and rounded cases
  // are covered.
  const vector<float> inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  const int N {static_cast<int>(inputs.size())};

  Array<float> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_exponential<float><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(h_output[i], expf(inputs[i])) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetExponentialTests, DoubleMatchesHostExp)
{
  const vector<double> inputs {0.0, 1.0, -1.0, 2.0, -2.0, 3.0};
  const int N {static_cast<int>(inputs.size())};

  Array<double> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_exponential<double><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<double> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_DOUBLE_EQ(h_output[i], std::exp(inputs[i])) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
// Input values {0, 1, -1, 2, -2} are exactly representable in half precision,
// so there is no input-rounding error. The only error is hexp's own rounding
// (≤0.5 ULP in half). Tolerance 2e-3 is ~2 ULP at these magnitudes.
//------------------------------------------------------------------------------
TEST(GetExponentialTests, HalfMatchesExpfWithinHalfPrecision)
{
  const vector<float> float_inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__half> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2half(float_inputs[i]);
  }

  Array<__half> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_exponential<__half><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__half> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    const float got {__half2float(h_output[i])};
    const float ref {expf(float_inputs[i])};
    EXPECT_NEAR(got, ref, 2e-3f) << "input=" << float_inputs[i];
  }
}

//------------------------------------------------------------------------------
// __half2 packs two __half values into one 32-bit register. h2exp computes
// exp on both lanes in a single instruction. Each lane is 16-bit, so the same
// 2e-3 tolerance as the scalar __half test applies per lane.
//------------------------------------------------------------------------------
TEST(GetExponentialTests, Half2EachLaneMatchesExpf)
{
  // lane0 and lane1 are independent inputs — use values exact in half.
  const vector<float> lane0 {0.0f,  1.0f, -1.0f};
  const vector<float> lane1 {2.0f, -2.0f,  0.5f};
  const int N {static_cast<int>(lane0.size())};

  vector<__half2> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __floats2half2_rn(lane0[i], lane1[i]);
  }

  Array<__half2> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_exponential<__half2><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__half2> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    const float got0 {__low2float(h_output[i])};
    const float got1 {__high2float(h_output[i])};
    EXPECT_NEAR(got0, expf(lane0[i]), 2e-3f) << "lane0 i=" << i;
    EXPECT_NEAR(got1, expf(lane1[i]), 2e-3f) << "lane1 i=" << i;
  }
}

//------------------------------------------------------------------------------
// bfloat16: 1 sign + 8 exponent + 7 mantissa bits — float's exponent range
// at ~2^-8 relative precision. Inputs {0, ±1, ±2} are exactly representable;
// hexp's bf16 result carries a few ULP, so tolerance is 2e-2·(1 + |ref|).
// Requires sm_80+ (this build targets sm_86 only).
//------------------------------------------------------------------------------
TEST(GetExponentialTests, Bfloat16MatchesExpfWithinBf16Precision)
{
  const vector<float> float_inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__nv_bfloat16> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2bfloat16(float_inputs[i]);
  }

  Array<__nv_bfloat16> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_exponential<__nv_bfloat16><<<1, 32>>>(
    d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__nv_bfloat16> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    const float got {__bfloat162float(h_output[i])};
    const float ref {expf(float_inputs[i])};
    EXPECT_NEAR(got, ref, 2e-2f * (1.0f + std::abs(ref)))
      << "input=" << float_inputs[i];
  }
}

//------------------------------------------------------------------------------
// get_approximate_exponential tests
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// __expf has max error ≤ 2 ULP vs the true value; host expf is ≤1 ULP.
// Combined difference is ≤3 ULP. 1 ULP at e ≈ 2.718 is ~2.4e-7.
// Tolerance 1e-5 catches wrong implementations while accommodating 2 ULP error.
//------------------------------------------------------------------------------
TEST(GetApproximateExponentialTests, FloatWithin2ULPOfHostExpf)
{
  const vector<float> inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  const int N {static_cast<int>(inputs.size())};

  Array<float> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_approximate_exponential<float><<<1, 32>>>(
    d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_NEAR(h_output[i], expf(inputs[i]), 1e-5f) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
// get_approximate_exponential<double> falls back to exp() — no CUDA double
// intrinsic. Result must be bitwise identical to get_exponential<double>.
//------------------------------------------------------------------------------
TEST(GetApproximateExponentialTests, DoubleIdenticalToGetExponentialDouble)
{
  const vector<double> inputs {0.0, 1.0, -1.0, 2.0, -2.0};
  const int N {static_cast<int>(inputs.size())};

  Array<double> d_input(N), d_approx(N), d_exact(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_approximate_exponential<double><<<1, 32>>>(
    d_approx.elements_, d_input.elements_, N);
  apply_get_exponential<double><<<1, 32>>>(d_exact.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<double> h_approx(N), h_exact(N);
  d_approx.copy_device_output_to_host(h_approx);
  d_exact.copy_device_output_to_host(h_exact);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_approx[i], h_exact[i]) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
// get_approximate_exponential<__half> uses hexp — the same instruction as
// get_exponential<__half>. Results must be bitwise identical.
//------------------------------------------------------------------------------
TEST(GetApproximateExponentialTests, HalfIdenticalToGetExponentialHalf)
{
  const vector<float> float_inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__half> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2half(float_inputs[i]);
  }

  Array<__half> d_input(N), d_approx(N), d_exact(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_approximate_exponential<__half><<<1, 32>>>(
    d_approx.elements_, d_input.elements_, N);
  apply_get_exponential<__half><<<1, 32>>>(d_exact.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__half> h_approx(N), h_exact(N);
  d_approx.copy_device_output_to_host(h_approx);
  d_exact.copy_device_output_to_host(h_exact);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(__half2float(h_approx[i]), __half2float(h_exact[i]))
      << "input=" << float_inputs[i];
  }
}

//------------------------------------------------------------------------------
// Both get_approximate_exponential<__half2> and get_exponential<__half2> call
// h2exp — no approximate/exact distinction at half precision. Results must be
// bitwise identical per lane.
//------------------------------------------------------------------------------
TEST(GetApproximateExponentialTests, Half2IdenticalToGetExponentialHalf2)
{
  const vector<float> lane0 {0.0f, 1.0f, -1.0f};
  const vector<float> lane1 {2.0f, -2.0f, 0.5f};
  const int N {static_cast<int>(lane0.size())};

  vector<__half2> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __floats2half2_rn(lane0[i], lane1[i]);
  }

  Array<__half2> d_input(N), d_approx(N), d_exact(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_approximate_exponential<__half2><<<1, 32>>>(
    d_approx.elements_, d_input.elements_, N);
  apply_get_exponential<__half2><<<1, 32>>>(d_exact.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__half2> h_approx(N), h_exact(N);
  d_approx.copy_device_output_to_host(h_approx);
  d_exact.copy_device_output_to_host(h_exact);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(__low2float(h_approx[i]),  __low2float(h_exact[i]))  << "lane0 i=" << i;
    EXPECT_EQ(__high2float(h_approx[i]), __high2float(h_exact[i])) << "lane1 i=" << i;
  }
}

//------------------------------------------------------------------------------
// Both bf16 exponentials call the same hexp overload — bitwise identical.
//------------------------------------------------------------------------------
TEST(GetApproximateExponentialTests, Bfloat16IdenticalToGetExponential)
{
  const vector<float> float_inputs {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__nv_bfloat16> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2bfloat16(float_inputs[i]);
  }

  Array<__nv_bfloat16> d_input(N), d_approx(N), d_exact(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_approximate_exponential<__nv_bfloat16><<<1, 32>>>(
    d_approx.elements_, d_input.elements_, N);
  apply_get_exponential<__nv_bfloat16><<<1, 32>>>(
    d_exact.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__nv_bfloat16> h_approx(N), h_exact(N);
  d_approx.copy_device_output_to_host(h_approx);
  d_exact.copy_device_output_to_host(h_exact);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(__bfloat162float(h_approx[i]), __bfloat162float(h_exact[i]))
      << "input=" << float_inputs[i];
  }
}

//------------------------------------------------------------------------------
// get_max tests
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetMaxTests, FloatBasicComparisons)
{
  const vector<float> a_vals   {3.0f, -1.0f, 4.0f};
  const vector<float> b_vals   {5.0f,  2.0f, 4.0f};
  const vector<float> expected {5.0f,  2.0f, 4.0f};
  const int N {static_cast<int>(a_vals.size())};

  Array<float> d_a(N), d_b(N), d_output(N);
  d_a.copy_host_input_to_device(a_vals);
  d_b.copy_host_input_to_device(b_vals);
  apply_get_max<float><<<1, 32>>>(
    d_output.elements_, d_a.elements_, d_b.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], expected[i]) << "i=" << i;
  }
}

//------------------------------------------------------------------------------
// fmaxf treats NaN as missing data: fmaxf(NaN, x) = x, fmaxf(x, NaN) = x.
// See https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/
//     group__CUDA__MATH__SINGLE.html
//     "Determines the maximum numeric value of the arguments. Treats NaN
//      arguments as missing data."
//------------------------------------------------------------------------------
TEST(GetMaxTests, FloatNaNTreatedAsMissingData)
{
  const float nan_val {std::numeric_limits<float>::quiet_NaN()};
  const vector<float> a_vals   {nan_val, 3.0f};
  const vector<float> b_vals   {5.0f,    nan_val};
  const vector<float> expected {5.0f,    3.0f};
  const int N {static_cast<int>(a_vals.size())};

  Array<float> d_a(N), d_b(N), d_output(N);
  d_a.copy_host_input_to_device(a_vals);
  d_b.copy_host_input_to_device(b_vals);
  apply_get_max<float><<<1, 32>>>(
    d_output.elements_, d_a.elements_, d_b.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], expected[i]) << "i=" << i;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetMaxTests, DoubleBasicComparisons)
{
  const vector<double> a_vals   {3.0, -1.0, 4.0};
  const vector<double> b_vals   {5.0,  2.0, 4.0};
  const vector<double> expected {5.0,  2.0, 4.0};
  const int N {static_cast<int>(a_vals.size())};

  Array<double> d_a(N), d_b(N), d_output(N);
  d_a.copy_host_input_to_device(a_vals);
  d_b.copy_host_input_to_device(b_vals);
  apply_get_max<double><<<1, 32>>>(
    d_output.elements_, d_a.elements_, d_b.elements_, N);
  cudaDeviceSynchronize();

  vector<double> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], expected[i]) << "i=" << i;
  }
}

//------------------------------------------------------------------------------
// get_sqrt tests
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Perfect squares are exactly representable in float; sqrtf of a perfect square
// is required to return the exact integer result (correctly rounded ≤0.5 ULP).
//------------------------------------------------------------------------------
TEST(GetSqrtTests, FloatPerfectSquares)
{
  const vector<float> inputs   {0.0f, 1.0f, 4.0f, 9.0f, 16.0f};
  const vector<float> expected {0.0f, 1.0f, 2.0f, 3.0f,  4.0f};
  const int N {static_cast<int>(inputs.size())};

  Array<float> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_sqrt<float><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(h_output[i], expected[i]) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetSqrtTests, FloatMatchesHostSqrtf)
{
  const vector<float> inputs {2.0f, 3.0f, 5.0f, 0.5f, 7.0f};
  const int N {static_cast<int>(inputs.size())};

  Array<float> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_sqrt<float><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<float> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(h_output[i], sqrtf(inputs[i])) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetSqrtTests, DoubleMatchesHostSqrt)
{
  const vector<double> inputs {0.0, 1.0, 2.0, 4.0, 9.0, 0.5};
  const int N {static_cast<int>(inputs.size())};

  Array<double> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(inputs);
  apply_get_sqrt<double><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<double> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_DOUBLE_EQ(h_output[i], std::sqrt(inputs[i])) << "input=" << inputs[i];
  }
}

//------------------------------------------------------------------------------
// Perfect squares that are exactly representable in half. hsqrt is correctly
// rounded (≤0.5 ULP), so integer square roots must be exact.
//------------------------------------------------------------------------------
TEST(GetSqrtTests, HalfPerfectSquaresExact)
{
  const vector<float> float_inputs {0.0f, 1.0f, 4.0f, 9.0f};
  const vector<float> expected     {0.0f, 1.0f, 2.0f, 3.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__half> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2half(float_inputs[i]);
  }

  Array<__half> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_sqrt<__half><<<1, 32>>>(d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__half> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(__half2float(h_output[i]), expected[i]) << "input=" << float_inputs[i];
  }
}

//------------------------------------------------------------------------------
// Small perfect squares are exactly representable in bfloat16 (7 mantissa
// bits cover integers to 256), and hsqrt is correctly rounded, so integer
// square roots must be exact.
//------------------------------------------------------------------------------
TEST(GetSqrtTests, Bfloat16PerfectSquaresExact)
{
  const vector<float> float_inputs {0.0f, 1.0f, 4.0f, 9.0f};
  const vector<float> expected     {0.0f, 1.0f, 2.0f, 3.0f};
  const int N {static_cast<int>(float_inputs.size())};

  vector<__nv_bfloat16> h_input(N);
  for (int i {0}; i < N; ++i)
  {
    h_input[i] = __float2bfloat16(float_inputs[i]);
  }

  Array<__nv_bfloat16> d_input(N), d_output(N);
  d_input.copy_host_input_to_device(h_input);
  apply_get_sqrt<__nv_bfloat16><<<1, 32>>>(
    d_output.elements_, d_input.elements_, N);
  cudaDeviceSynchronize();

  vector<__nv_bfloat16> h_output(N);
  d_output.copy_device_output_to_host(h_output);

  for (int i {0}; i < N; ++i)
  {
    EXPECT_EQ(__bfloat162float(h_output[i]), expected[i])
      << "input=" << float_inputs[i];
  }
}

} // namespace MathFunctions
} // namespace Numerics
} // namespace GoogleUnitTests
