#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using DataStructures::Array;
using std::vector;

namespace GoogleUnitTests
{
namespace LLM
{
namespace AttentionForward
{

#include "LLM/AttentionForward/FlashAttention.h"
#include "LLM/attention_forward.h"

using ::LLM::AttentionForward::flash_attention_forward;
using ::LLM::attention_query_key_kernel1;

//------------------------------------------------------------------------------
/// Reference: compute attention on CPU for correctness comparison.
/// Output shape: [B, NH, N, d]
/// Input Q,K,V shape: [B, NH, N, d] (already split into heads).
//------------------------------------------------------------------------------
static void reference_attention_cpu(
  vector<float>& output,
  const vector<float>& Q,
  const vector<float>& K,
  const vector<float>& V,
  const int B,
  const int NH,
  const int N,
  const int d)
{
  output.assign(B * NH * N * d, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  for (int b = 0; b < B; ++b)
  {
    for (int h = 0; h < NH; ++h)
    {
      const int bh = b * NH + h;
      const float* Q_bh = Q.data() + bh * N * d;
      const float* K_bh = K.data() + bh * N * d;
      const float* V_bh = V.data() + bh * N * d;
      float* O_bh = output.data() + bh * N * d;

      // S = Q K^T * scale  [N x N]
      vector<float> S(N * N);
      for (int i = 0; i < N; ++i)
      {
        for (int j = 0; j < N; ++j)
        {
          float dot = 0.0f;
          for (int k = 0; k < d; ++k)
          {
            dot += Q_bh[i * d + k] * K_bh[j * d + k];
          }
          S[i * N + j] = dot * scale;
        }
      }

      // P = softmax(S) row-wise  [N x N]
      vector<float> P(N * N);
      for (int i = 0; i < N; ++i)
      {
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j)
        {
          row_max = std::max(row_max, S[i * N + j]);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j)
        {
          P[i * N + j] = std::exp(S[i * N + j] - row_max);
          row_sum += P[i * N + j];
        }
        for (int j = 0; j < N; ++j)
        {
          P[i * N + j] /= row_sum;
        }
      }

      // O = P V  [N x d]
      for (int i = 0; i < N; ++i)
      {
        for (int k = 0; k < d; ++k)
        {
          float acc = 0.0f;
          for (int j = 0; j < N; ++j)
          {
            acc += P[i * N + j] * V_bh[j * d + k];
          }
          O_bh[i * d + k] = acc;
        }
      }
    }
  }
}

class FlashAttentionTests : public ::testing::Test
{
  protected:
    static constexpr int kHeadDim = 64;
    static constexpr int kBr = 32;
    static constexpr int kBc = 32;

    void fill_random(vector<float>& v, unsigned seed = 42)
    {
      std::mt19937 gen(seed);
      std::normal_distribution<float> dist(0.0f, 0.5f);
      for (auto& x : v) { x = dist(gen); }
    }

    float run_flash_and_compare(
      const int B, const int NH, const int N,
      const float tolerance = 1e-4f)
    {
      const int total = B * NH * N * kHeadDim;
      vector<float> Q(total), K(total), V(total);
      fill_random(Q, 1);
      fill_random(K, 2);
      fill_random(V, 3);

      // CPU reference
      vector<float> ref_out;
      reference_attention_cpu(ref_out, Q, K, V, B, NH, N, kHeadDim);

      // GPU FlashAttention
      Array<float> d_Q(total), d_K(total), d_V(total), d_O(total);
      d_Q.copy_host_input_to_device(Q);
      d_K.copy_host_input_to_device(K);
      d_V.copy_host_input_to_device(V);

      const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
      flash_attention_forward<float, kHeadDim, kBr, kBc>(
        d_O.elements_, d_Q.elements_, d_K.elements_, d_V.elements_,
        B, NH, N, scale);
      cudaDeviceSynchronize();

      vector<float> flash_out(total);
      d_O.copy_device_output_to_host(flash_out);

      // Compute max absolute error
      float max_err = 0.0f;
      for (int i = 0; i < total; ++i)
      {
        max_err = std::max(max_err, std::abs(flash_out[i] - ref_out[i]));
      }
      return max_err;
    }
};

// Verify output matches reference CPU attention for small sequence.
TEST_F(FlashAttentionTests, MatchesReferenceSmallSequence)
{
  // N=32 fits in a single row tile — no tile-boundary rescaling needed.
  const float max_err = run_flash_and_compare(1, 1, 32);
  EXPECT_LT(max_err, 1e-4f)
    << "Max absolute error vs CPU reference: " << max_err;
}

// N > kBr forces multiple row tiles — exercises the outer loop.
TEST_F(FlashAttentionTests, MatchesReferenceMultipleRowTiles)
{
  const float max_err = run_flash_and_compare(1, 1, 128);
  EXPECT_LT(max_err, 1e-4f)
    << "Max absolute error vs CPU reference: " << max_err;
}

// N > kBc forces multiple col-block iterations — exercises online rescaling.
TEST_F(FlashAttentionTests, MatchesReferenceMultipleColBlocks)
{
  const float max_err = run_flash_and_compare(1, 1, 64);
  EXPECT_LT(max_err, 1e-4f)
    << "Max absolute error vs CPU reference: " << max_err;
}

// Multi-head: results must be independent per head.
TEST_F(FlashAttentionTests, MultiHeadIndependence)
{
  const float max_err = run_flash_and_compare(1, 4, 64);
  EXPECT_LT(max_err, 1e-4f)
    << "Max absolute error vs CPU reference: " << max_err;
}

// Multi-batch.
TEST_F(FlashAttentionTests, MultiBatch)
{
  const float max_err = run_flash_and_compare(2, 2, 64);
  EXPECT_LT(max_err, 1e-4f)
    << "Max absolute error vs CPU reference: " << max_err;
}

// Numerically stable: large pre-attention scores should not produce NaN/Inf.
TEST_F(FlashAttentionTests, NumericalStabilityLargeScores)
{
  constexpr int B = 1, NH = 1, N = 32;
  const int total = B * NH * N * kHeadDim;

  // Large-magnitude Q produces large dot products before scaling.
  vector<float> Q(total, 10.0f), K(total, 10.0f), V(total, 1.0f);

  Array<float> d_Q(total), d_K(total), d_V(total), d_O(total);
  d_Q.copy_host_input_to_device(Q);
  d_K.copy_host_input_to_device(K);
  d_V.copy_host_input_to_device(V);

  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  flash_attention_forward<float, kHeadDim, kBr, kBc>(
    d_O.elements_, d_Q.elements_, d_K.elements_, d_V.elements_,
    B, NH, N, scale);
  cudaDeviceSynchronize();

  vector<float> flash_out(total);
  d_O.copy_device_output_to_host(flash_out);

  for (const float v : flash_out)
  {
    EXPECT_FALSE(std::isnan(v)) << "NaN in output";
    EXPECT_FALSE(std::isinf(v)) << "Inf in output";
  }
}

} // namespace AttentionForward
} // namespace LLM
} // namespace GoogleUnitTests
