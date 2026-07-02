#include "DataStructures/Array.h"
#include "Transformer/Attention/attention_scores.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::sqrt;
using std::vector;
using Transformer::Attention::attention_scores;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// CPU reference: S_ij = q_i · k_j / √d_k, accumulated in double.
//------------------------------------------------------------------------------
vector<float> attention_scores_cpu(
  const vector<float>& queries,
  const vector<float>& keys,
  const int n,
  const int d_k)
{
  vector<float> scores(n * n);
  const double scale {1.0 / sqrt(static_cast<double>(d_k))};
  for (int i {0}; i < n; ++i)
  {
    for (int j {0}; j < n; ++j)
    {
      double dot {0.0};
      for (int d {0}; d < d_k; ++d)
      {
        dot += static_cast<double>(queries[i * d_k + d]) *
          static_cast<double>(keys[j * d_k + d]);
      }
      scores[i * n + j] = static_cast<float>(dot * scale);
    }
  }
  return scores;
}

//------------------------------------------------------------------------------
// Q rows and K rows are standard basis vectors: q_i · k_j = δ_ij, so
// S = I / √d_k. Hand-checkable smallest interesting case.
//------------------------------------------------------------------------------
TEST(AttentionScoresTests, OrthonormalRowsGiveScaledIdentity)
{
  constexpr int n {4};
  constexpr int kHD {4};

  // Row i of Q and of K is e_i (one-hot).
  vector<float> queries(n * kHD, 0.0f);
  vector<float> keys(n * kHD, 0.0f);
  for (int i {0}; i < n; ++i)
  {
    queries[i * kHD + i] = 1.0f;
    keys[i * kHD + i] = 1.0f;
  }

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_scores(n * n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);

  attention_scores<float, kHD><<<n, 128>>>(
    d_scores.elements_, d_queries.elements_, d_keys.elements_, n);
  cudaDeviceSynchronize();

  vector<float> scores(n * n);
  d_scores.copy_device_output_to_host(scores);

  // √d_k = √4 = 2, so diagonal entries are 1/2 and off-diagonal are 0.
  for (int i {0}; i < n; ++i)
  {
    for (int j {0}; j < n; ++j)
    {
      const float expected {(i == j) ? 0.5f : 0.0f};
      EXPECT_NEAR(scores[i * n + j], expected, 1e-6f)
        << "entry (" << i << ", " << j << ")";
    }
  }
}

//------------------------------------------------------------------------------
// Hand-computed 2×2 case with d_k = 4 (√d_k = 2):
//   q_0 = [1,2,3,4], q_1 = [0,1,0,1]
//   k_0 = [1,1,1,1], k_1 = [1,0,-1,0]
//   S_00 = (1+2+3+4)/2 = 5,     S_01 = (1-3)/2 = -1
//   S_10 = (1+1)/2 = 1,         S_11 = 0/2 = 0
//------------------------------------------------------------------------------
TEST(AttentionScoresTests, HandComputedTwoByTwo)
{
  constexpr int n {2};
  constexpr int kHD {4};

  const vector<float> queries {
    1.0f, 2.0f, 3.0f, 4.0f,
    0.0f, 1.0f, 0.0f, 1.0f};
  const vector<float> keys {
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 0.0f, -1.0f, 0.0f};

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_scores(n * n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);

  attention_scores<float, kHD><<<n, 128>>>(
    d_scores.elements_, d_queries.elements_, d_keys.elements_, n);
  cudaDeviceSynchronize();

  vector<float> scores(n * n);
  d_scores.copy_device_output_to_host(scores);

  EXPECT_NEAR(scores[0], 5.0f, 1e-6f);
  EXPECT_NEAR(scores[1], -1.0f, 1e-6f);
  EXPECT_NEAR(scores[2], 1.0f, 1e-6f);
  EXPECT_NEAR(scores[3], 0.0f, 1e-6f);
}

//------------------------------------------------------------------------------
// n larger than blockDim exercises the strided key loop (each thread owns
// multiple key columns); kHD larger than one warp exercises the cooperative
// shared-memory load of q_i. Compare against the CPU reference.
//------------------------------------------------------------------------------
TEST(AttentionScoresTests, MatchesCpuReferenceOnLargerInputs)
{
  constexpr int n {200};
  constexpr int kHD {64};

  vector<float> queries(n * kHD);
  vector<float> keys(n * kHD);
  // Deterministic, sign-alternating values in [-1, 1] so dot products stay
  // O(1) and float vs. double accumulation differences stay small.
  for (int i {0}; i < n * kHD; ++i)
  {
    queries[i] = 0.02f * static_cast<float>((i * 7 + 3) % 101 - 50) / 50.0f;
    keys[i] = 0.02f * static_cast<float>((i * 13 + 5) % 89 - 44) / 44.0f;
  }

  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_scores(n * n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);

  attention_scores<float, kHD><<<n, 128>>>(
    d_scores.elements_, d_queries.elements_, d_keys.elements_, n);
  cudaDeviceSynchronize();

  vector<float> scores(n * n);
  d_scores.copy_device_output_to_host(scores);

  const vector<float> expected {attention_scores_cpu(queries, keys, n, kHD)};
  for (int i {0}; i < n * n; ++i)
  {
    ASSERT_NEAR(scores[i], expected[i], 1e-5f) << "index " << i;
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
