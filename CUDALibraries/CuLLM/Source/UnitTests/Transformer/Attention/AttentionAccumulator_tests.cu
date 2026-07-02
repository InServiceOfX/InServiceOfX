#include "DataStructures/Array.h"
#include "Numerics/Constants/get_infinity.h"
#include "Transformer/Attention/AttentionAccumulator.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;
using Transformer::Attention::AttentionAccumulator;
using Transformer::Attention::attention_identity;
using Transformer::Attention::merge;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

// All tests use kHD=4 so values are readable in the expected output.
constexpr int kHD {4};
using Acc4 = AttentionAccumulator<float, kHD>;

//------------------------------------------------------------------------------
// Device helpers
//------------------------------------------------------------------------------

// Single-element accumulator: score s, V row is a one-hot unit vector
// (1 at position hot, 0 elsewhere).  Represents one key-value position with
// max_value = s, sum = exp(s-s) = 1, output[hot] = exp(s-s)*1 = 1.
__device__ Acc4 make_single(const float score, const int hot)
{
  Acc4 acc;
  acc.max_value = score;
  acc.sum = 1.0f;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    acc.output[d] = (d == hot) ? 1.0f : 0.0f;
  }
  return acc;
}

//------------------------------------------------------------------------------
// Kernels
//------------------------------------------------------------------------------

// Test that merge(identity, acc) and merge(acc, identity) both equal acc.
// acc: max=2, sum=3, output=[1,2,3,4].
// Writes two result sets side-by-side in out_max[0..1], out_sum[0..1],
// out_output[0..2*kHD-1].
__global__ void kernel_identity_merge(
  float* out_max, float* out_sum, float* out_output)
{
  Acc4 acc;
  acc.max_value = 2.0f;
  acc.sum = 3.0f;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    acc.output[d] = static_cast<float>(d + 1);
  }

  const Acc4 id {attention_identity<float, kHD>()};
  const Acc4 r0 {merge(id, acc)};  // identity on left
  const Acc4 r1 {merge(acc, id)};  // identity on right

  out_max[0] = r0.max_value;
  out_max[1] = r1.max_value;
  out_sum[0] = r0.sum;
  out_sum[1] = r1.sum;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    out_output[0 * kHD + d] = r0.output[d];
    out_output[1 * kHD + d] = r1.output[d];
  }
}

// Merge two single-element accumulators:
//   a: s_0=1, V_0=[1,0,0,0]   → (1, 1, [1,0,0,0])
//   b: s_1=2, V_1=[0,1,0,0]   → (2, 1, [0,1,0,0])
// Expected: m=2, ℓ=exp(-1)+1, õ=[exp(-1), 1, 0, 0].
__global__ void kernel_two_element_merge(
  float* out_max, float* out_sum, float* out_output)
{
  const Acc4 a {make_single(1.0f, 0)};
  const Acc4 b {make_single(2.0f, 1)};
  const Acc4 r {merge(a, b)};

  *out_max = r.max_value;
  *out_sum = r.sum;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    out_output[d] = r.output[d];
  }
}

// Verify merge is commutative: merge(a,b) == merge(b,a).
// Writes (max, sum, output[0..kHD-1]) for each ordering.
__global__ void kernel_commutativity(float* out_ab, float* out_ba)
{
  const Acc4 a {make_single(1.5f, 0)};
  const Acc4 b {make_single(3.0f, 2)};
  const Acc4 ab {merge(a, b)};
  const Acc4 ba {merge(b, a)};

  out_ab[0] = ab.max_value;
  out_ab[1] = ab.sum;
  out_ba[0] = ba.max_value;
  out_ba[1] = ba.sum;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    out_ab[2 + d] = ab.output[d];
    out_ba[2 + d] = ba.output[d];
  }
}

// Sequential fold over 4 positions with s=[1,2,3,4] and V=I_4.
// After the fold, normalized output o = õ/ℓ must equal softmax([1,2,3,4]).
__global__ void kernel_sequential_fold(float* out_output, float* out_sum)
{
  Acc4 acc {attention_identity<float, kHD>()};
  acc = merge(acc, make_single(1.0f, 0));
  acc = merge(acc, make_single(2.0f, 1));
  acc = merge(acc, make_single(3.0f, 2));
  acc = merge(acc, make_single(4.0f, 3));

  *out_sum = acc.sum;
  #pragma unroll
  for (int d {0}; d < kHD; ++d)
  {
    out_output[d] = acc.output[d] / acc.sum;
  }
}

//------------------------------------------------------------------------------
// Tests
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionAccumulatorTests, IdentityMerge)
{
  // merge(identity, acc) and merge(acc, identity) must both return acc unchanged.
  Array<float> d_max(2), d_sum(2), d_output(2 * kHD);
  kernel_identity_merge<<<1, 1>>>(
    d_max.elements_, d_sum.elements_, d_output.elements_);
  cudaDeviceSynchronize();

  vector<float> h_max(2), h_sum(2), h_output(2 * kHD);
  d_max.copy_device_output_to_host(h_max);
  d_sum.copy_device_output_to_host(h_sum);
  d_output.copy_device_output_to_host(h_output);

  for (int order {0}; order < 2; ++order)
  {
    EXPECT_EQ(h_max[order], 2.0f) << "order=" << order;
    EXPECT_EQ(h_sum[order], 3.0f) << "order=" << order;
    for (int d {0}; d < kHD; ++d)
    {
      EXPECT_EQ(h_output[order * kHD + d], static_cast<float>(d + 1))
        << "order=" << order << " d=" << d;
    }
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionAccumulatorTests, TwoSingleElementMerge)
{
  // a=(1,1,[1,0,0,0]), b=(2,1,[0,1,0,0]).
  // Expected: m=2, ℓ=exp(-1)+1, õ=[exp(-1),1,0,0].
  Array<float> d_max(1), d_sum(1), d_output(kHD);
  kernel_two_element_merge<<<1, 1>>>(
    d_max.elements_, d_sum.elements_, d_output.elements_);
  cudaDeviceSynchronize();

  vector<float> h_max(1), h_sum(1), h_output(kHD);
  d_max.copy_device_output_to_host(h_max);
  d_sum.copy_device_output_to_host(h_sum);
  d_output.copy_device_output_to_host(h_output);

  const float e_neg1 {exp(-1.0f)};
  EXPECT_EQ(h_max[0], 2.0f);
  EXPECT_NEAR(h_sum[0],    e_neg1 + 1.0f, 1e-6f);
  EXPECT_NEAR(h_output[0], e_neg1,         1e-6f);
  EXPECT_NEAR(h_output[1], 1.0f,           1e-6f);
  EXPECT_NEAR(h_output[2], 0.0f,           1e-6f);
  EXPECT_NEAR(h_output[3], 0.0f,           1e-6f);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionAccumulatorTests, CommutativityCheck)
{
  // merge(a, b) must equal merge(b, a) in all fields:
  // max_value, sum, and every component of output[].
  Array<float> d_ab(2 + kHD), d_ba(2 + kHD);
  kernel_commutativity<<<1, 1>>>(d_ab.elements_, d_ba.elements_);
  cudaDeviceSynchronize();

  vector<float> h_ab(2 + kHD), h_ba(2 + kHD);
  d_ab.copy_device_output_to_host(h_ab);
  d_ba.copy_device_output_to_host(h_ba);

  for (int i {0}; i < 2 + kHD; ++i)
  {
    EXPECT_NEAR(h_ab[i], h_ba[i], 1e-6f) << "index=" << i;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionAccumulatorTests, SequentialFoldMatchesReference)
{
  // Fold single-element accumulators for s=[1,2,3,4], V=I_4 (identity matrix).
  // Because V=I_4, softmax(s)·V = softmax(s), so the normalized output
  // o = õ/ℓ must equal softmax([1,2,3,4]).
  Array<float> d_output(kHD), d_sum(1);
  kernel_sequential_fold<<<1, 1>>>(d_output.elements_, d_sum.elements_);
  cudaDeviceSynchronize();

  vector<float> h_output(kHD), h_sum(1);
  d_output.copy_device_output_to_host(h_output);
  d_sum.copy_device_output_to_host(h_sum);

  // CPU reference: softmax([1,2,3,4]), max=4
  const float max_val {4.0f};
  const float e0 {exp(1.0f - max_val)};
  const float e1 {exp(2.0f - max_val)};
  const float e2 {exp(3.0f - max_val)};
  const float e3 {exp(4.0f - max_val)};
  const float Z  {e0 + e1 + e2 + e3};

  EXPECT_NEAR(h_sum[0],    Z,        1e-5f);
  EXPECT_NEAR(h_output[0], e0 / Z,   1e-5f);
  EXPECT_NEAR(h_output[1], e1 / Z,   1e-5f);
  EXPECT_NEAR(h_output[2], e2 / Z,   1e-5f);
  EXPECT_NEAR(h_output[3], e3 / Z,   1e-5f);
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
