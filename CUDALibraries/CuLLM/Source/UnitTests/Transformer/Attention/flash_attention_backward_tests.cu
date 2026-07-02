#include "DataStructures/Array.h"
#include "Transformer/Attention/flash_attention_backward.h"
#include "Transformer/Attention/flash_attention_warp_cooperative.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::sqrt;
using std::vector;
using Transformer::Attention::flash_attention_backward;
using Transformer::Attention::flash_attention_warp_cooperative;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace Attention
{

//------------------------------------------------------------------------------
// Double-precision CPU forward: O = softmax(QK^⊤/√d)V, optionally causal.
// Used both as the finite-difference oracle and inside the reference
// backward below.
//------------------------------------------------------------------------------
vector<double> attention_forward_cpu_double(
  const vector<double>& queries,
  const vector<double>& keys,
  const vector<double>& values,
  const int n,
  const int d,
  const bool causal)
{
  vector<double> output(n * d, 0.0);
  const double scale {1.0 / sqrt(static_cast<double>(d))};

  for (int i {0}; i < n; ++i)
  {
    const int limit {causal ? i + 1 : n};
    vector<double> scores(limit);
    for (int j {0}; j < limit; ++j)
    {
      double dot {0.0};
      for (int c {0}; c < d; ++c)
      {
        dot += queries[i * d + c] * keys[j * d + c];
      }
      scores[j] = dot * scale;
    }
    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    for (int j {0}; j < limit; ++j)
    {
      sum += exp(scores[j] - max_score);
    }
    for (int a {0}; a < d; ++a)
    {
      double accumulated {0.0};
      for (int j {0}; j < limit; ++j)
      {
        accumulated += (exp(scores[j] - max_score) / sum) * values[j * d + a];
      }
      output[i * d + a] = accumulated;
    }
  }
  return output;
}

struct CpuGradients
{
  vector<float> queries;
  vector<float> keys;
  vector<float> values;
};

//------------------------------------------------------------------------------
// Double-precision CPU backward implementing the gradient formulas from the
// section on Gradients of Scaled Dot-Product Attention in
// FlashAttention.tex: dP = dO V^⊤, D_i = Σ_k P_ik dP_ik,
// dS = P ⊙ (dP − D 1^⊤), dQ = dS K/√d, dK = dS^⊤ Q/√d, dV = P^⊤ dO.
//------------------------------------------------------------------------------
CpuGradients attention_backward_cpu(
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const vector<float>& gradient_output,
  const int n,
  const int d,
  const bool causal)
{
  const double scale {1.0 / sqrt(static_cast<double>(d))};

  // P (row-wise safe softmax; masked entries are exactly 0).
  vector<double> weights(n * n, 0.0);
  for (int i {0}; i < n; ++i)
  {
    const int limit {causal ? i + 1 : n};
    vector<double> scores(limit);
    for (int j {0}; j < limit; ++j)
    {
      double dot {0.0};
      for (int c {0}; c < d; ++c)
      {
        dot += static_cast<double>(queries[i * d + c]) *
          static_cast<double>(keys[j * d + c]);
      }
      scores[j] = dot * scale;
    }
    const double max_score {*std::max_element(scores.begin(), scores.end())};
    double sum {0.0};
    for (int j {0}; j < limit; ++j)
    {
      sum += exp(scores[j] - max_score);
    }
    for (int j {0}; j < limit; ++j)
    {
      weights[i * n + j] = exp(scores[j] - max_score) / sum;
    }
  }

  // dP_ij = ⟨dO_i, v_j⟩ and the row scalars D_i = Σ_k P_ik dP_ik.
  vector<double> weight_gradients(n * n);
  vector<double> row_dots(n, 0.0);
  for (int i {0}; i < n; ++i)
  {
    for (int j {0}; j < n; ++j)
    {
      double dot {0.0};
      for (int a {0}; a < d; ++a)
      {
        dot += static_cast<double>(gradient_output[i * d + a]) *
          static_cast<double>(values[j * d + a]);
      }
      weight_gradients[i * n + j] = dot;
      row_dots[i] += weights[i * n + j] * dot;
    }
  }

  CpuGradients gradients;
  gradients.queries.assign(n * d, 0.0f);
  gradients.keys.assign(n * d, 0.0f);
  gradients.values.assign(n * d, 0.0f);

  for (int i {0}; i < n; ++i)
  {
    for (int j {0}; j < n; ++j)
    {
      const double score_gradient {
        weights[i * n + j] * (weight_gradients[i * n + j] - row_dots[i])};
      for (int b {0}; b < d; ++b)
      {
        gradients.queries[i * d + b] += static_cast<float>(
          score_gradient * static_cast<double>(keys[j * d + b]) * scale);
        gradients.keys[j * d + b] += static_cast<float>(
          score_gradient * static_cast<double>(queries[i * d + b]) * scale);
      }
      for (int a {0}; a < d; ++a)
      {
        gradients.values[j * d + a] += static_cast<float>(
          weights[i * n + j] *
            static_cast<double>(gradient_output[i * d + a]));
      }
    }
  }
  return gradients;
}

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_backward_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 105.0f;
  }
  return result;
}

//------------------------------------------------------------------------------
// Runs GPU forward (for O and the logsumexp L) then the backward kernels.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, bool kCausal>
CpuGradients run_backward(
  const vector<float>& queries,
  const vector<float>& keys,
  const vector<float>& values,
  const vector<float>& gradient_output,
  const int n)
{
  Array<float> d_queries(n * kHD);
  Array<float> d_keys(n * kHD);
  Array<float> d_values(n * kHD);
  Array<float> d_output(n * kHD);
  Array<float> d_logsumexp(n);
  Array<float> d_gradient_output(n * kHD);
  Array<float> d_gradient_queries(n * kHD);
  Array<float> d_gradient_keys(n * kHD);
  Array<float> d_gradient_values(n * kHD);
  Array<float> d_row_dots(n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);
  d_gradient_output.copy_host_input_to_device(gradient_output);

  flash_attention_warp_cooperative<float, kHD, kWarps, kCausal>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n);

  flash_attention_backward<float, kHD, kWarps, kCausal>(
    d_gradient_queries.elements_,
    d_gradient_keys.elements_,
    d_gradient_values.elements_,
    d_row_dots.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    d_output.elements_,
    d_gradient_output.elements_,
    d_logsumexp.elements_,
    n);
  cudaDeviceSynchronize();

  CpuGradients gradients;
  gradients.queries.resize(n * kHD);
  gradients.keys.resize(n * kHD);
  gradients.values.resize(n * kHD);
  d_gradient_queries.copy_device_output_to_host(gradients.queries);
  d_gradient_keys.copy_device_output_to_host(gradients.keys);
  d_gradient_values.copy_device_output_to_host(gradients.values);
  return gradients;
}

//------------------------------------------------------------------------------
// Full backward against the analytic double-precision CPU reference, on a
// size exercising partial tiles in both passes.
//------------------------------------------------------------------------------
TEST(FlashAttentionBackwardTests, MatchesCpuReference)
{
  constexpr int n {100};
  constexpr int kHD {32};

  const vector<float> queries {make_backward_inputs(n * kHD, 3)};
  const vector<float> keys {make_backward_inputs(n * kHD, 5)};
  const vector<float> values {make_backward_inputs(n * kHD, 11)};
  const vector<float> gradient_output {make_backward_inputs(n * kHD, 7)};

  const CpuGradients actual {run_backward<kHD, 4, false>(
    queries, keys, values, gradient_output, n)};
  const CpuGradients expected {attention_backward_cpu(
    queries, keys, values, gradient_output, n, kHD, false)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(actual.queries[i], expected.queries[i], 2e-4f)
      << "dQ index " << i;
    ASSERT_NEAR(actual.keys[i], expected.keys[i], 2e-4f)
      << "dK index " << i;
    ASSERT_NEAR(actual.values[i], expected.values[i], 2e-4f)
      << "dV index " << i;
  }
}

//------------------------------------------------------------------------------
// Causal backward: gradients restrict to j ≤ i (dS_ij = 0 wherever
// P_ij = 0), and both passes skip tiles on their respective sides of the
// diagonal. Same CPU reference with the causal flag.
//------------------------------------------------------------------------------
TEST(FlashAttentionBackwardTests, CausalMatchesCpuReference)
{
  constexpr int n {100};
  constexpr int kHD {32};

  const vector<float> queries {make_backward_inputs(n * kHD, 13)};
  const vector<float> keys {make_backward_inputs(n * kHD, 17)};
  const vector<float> values {make_backward_inputs(n * kHD, 19)};
  const vector<float> gradient_output {make_backward_inputs(n * kHD, 23)};

  const CpuGradients actual {run_backward<kHD, 4, true>(
    queries, keys, values, gradient_output, n)};
  const CpuGradients expected {attention_backward_cpu(
    queries, keys, values, gradient_output, n, kHD, true)};

  for (int i {0}; i < n * kHD; ++i)
  {
    ASSERT_NEAR(actual.queries[i], expected.queries[i], 2e-4f)
      << "dQ index " << i;
    ASSERT_NEAR(actual.keys[i], expected.keys[i], 2e-4f)
      << "dK index " << i;
    ASSERT_NEAR(actual.values[i], expected.values[i], 2e-4f)
      << "dV index " << i;
  }
}

//------------------------------------------------------------------------------
// Finite-difference check: for the scalar loss φ = Σ_{ia} W_ia O_ia (so
// dO = W), central differences of the double-precision CPU forward must
// match every kernel gradient entry. This validates the *derivation* end to
// end, independently of the analytic CPU backward.
//------------------------------------------------------------------------------
TEST(FlashAttentionBackwardTests, MatchesFiniteDifferences)
{
  constexpr int n {8};
  constexpr int kHD {32};
  constexpr double epsilon {1e-4};

  const vector<float> queries {make_backward_inputs(n * kHD, 3)};
  const vector<float> keys {make_backward_inputs(n * kHD, 5)};
  const vector<float> values {make_backward_inputs(n * kHD, 11)};
  const vector<float> loss_weights {make_backward_inputs(n * kHD, 29)};

  const CpuGradients actual {run_backward<kHD, 4, false>(
    queries, keys, values, loss_weights, n)};

  const vector<double> base_queries(queries.begin(), queries.end());
  const vector<double> base_keys(keys.begin(), keys.end());
  const vector<double> base_values(values.begin(), values.end());

  const auto loss = [&](
    const vector<double>& q,
    const vector<double>& k,
    const vector<double>& v) -> double
  {
    const vector<double> output {
      attention_forward_cpu_double(q, k, v, n, kHD, false)};
    double total {0.0};
    for (int i {0}; i < n * kHD; ++i)
    {
      total += static_cast<double>(loss_weights[i]) * output[i];
    }
    return total;
  };

  for (int i {0}; i < n * kHD; ++i)
  {
    // dQ entry i.
    vector<double> perturbed {base_queries};
    perturbed[i] = base_queries[i] + epsilon;
    const double loss_plus_q {loss(perturbed, base_keys, base_values)};
    perturbed[i] = base_queries[i] - epsilon;
    const double loss_minus_q {loss(perturbed, base_keys, base_values)};
    ASSERT_NEAR(
      actual.queries[i],
      static_cast<float>((loss_plus_q - loss_minus_q) / (2.0 * epsilon)),
      2e-4f) << "dQ index " << i;

    // dK entry i.
    perturbed = base_keys;
    perturbed[i] = base_keys[i] + epsilon;
    const double loss_plus_k {loss(base_queries, perturbed, base_values)};
    perturbed[i] = base_keys[i] - epsilon;
    const double loss_minus_k {loss(base_queries, perturbed, base_values)};
    ASSERT_NEAR(
      actual.keys[i],
      static_cast<float>((loss_plus_k - loss_minus_k) / (2.0 * epsilon)),
      2e-4f) << "dK index " << i;

    // dV entry i.
    perturbed = base_values;
    perturbed[i] = base_values[i] + epsilon;
    const double loss_plus_v {loss(base_queries, base_keys, perturbed)};
    perturbed[i] = base_values[i] - epsilon;
    const double loss_minus_v {loss(base_queries, base_keys, perturbed)};
    ASSERT_NEAR(
      actual.values[i],
      static_cast<float>((loss_plus_v - loss_minus_v) / (2.0 * epsilon)),
      2e-4f) << "dV index " << i;
  }
}

//------------------------------------------------------------------------------
// Batched slices are independent in the backward as well: a batched launch
// must equal per-slice launches bit for bit.
//------------------------------------------------------------------------------
TEST(FlashAttentionBackwardTests, BatchedMatchesPerSlice)
{
  constexpr int batch_heads {3};
  constexpr int n {64};
  constexpr int kHD {32};
  constexpr int slice {n * kHD};

  const vector<float> queries {make_backward_inputs(batch_heads * slice, 3)};
  const vector<float> keys {make_backward_inputs(batch_heads * slice, 5)};
  const vector<float> values {make_backward_inputs(batch_heads * slice, 11)};
  const vector<float> gradient_output {
    make_backward_inputs(batch_heads * slice, 7)};

  Array<float> d_queries(batch_heads * slice);
  Array<float> d_keys(batch_heads * slice);
  Array<float> d_values(batch_heads * slice);
  Array<float> d_output(batch_heads * slice);
  Array<float> d_logsumexp(batch_heads * n);
  Array<float> d_gradient_output(batch_heads * slice);
  Array<float> d_gradient_queries(batch_heads * slice);
  Array<float> d_gradient_keys(batch_heads * slice);
  Array<float> d_gradient_values(batch_heads * slice);
  Array<float> d_row_dots(batch_heads * n);
  d_queries.copy_host_input_to_device(queries);
  d_keys.copy_host_input_to_device(keys);
  d_values.copy_host_input_to_device(values);
  d_gradient_output.copy_host_input_to_device(gradient_output);

  flash_attention_warp_cooperative<float, kHD, 4>(
    d_output.elements_,
    d_logsumexp.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    n,
    batch_heads);
  flash_attention_backward<float, kHD, 4>(
    d_gradient_queries.elements_,
    d_gradient_keys.elements_,
    d_gradient_values.elements_,
    d_row_dots.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    d_output.elements_,
    d_gradient_output.elements_,
    d_logsumexp.elements_,
    n,
    batch_heads);
  cudaDeviceSynchronize();

  vector<float> batched_gradient_queries(batch_heads * slice);
  vector<float> batched_gradient_keys(batch_heads * slice);
  vector<float> batched_gradient_values(batch_heads * slice);
  d_gradient_queries.copy_device_output_to_host(batched_gradient_queries);
  d_gradient_keys.copy_device_output_to_host(batched_gradient_keys);
  d_gradient_values.copy_device_output_to_host(batched_gradient_values);

  for (int s {0}; s < batch_heads; ++s)
  {
    const vector<float> slice_queries(
      queries.begin() + s * slice, queries.begin() + (s + 1) * slice);
    const vector<float> slice_keys(
      keys.begin() + s * slice, keys.begin() + (s + 1) * slice);
    const vector<float> slice_values(
      values.begin() + s * slice, values.begin() + (s + 1) * slice);
    const vector<float> slice_gradient_output(
      gradient_output.begin() + s * slice,
      gradient_output.begin() + (s + 1) * slice);

    const CpuGradients single {run_backward<kHD, 4, false>(
      slice_queries, slice_keys, slice_values, slice_gradient_output, n)};

    for (int i {0}; i < slice; ++i)
    {
      ASSERT_EQ(batched_gradient_queries[s * slice + i], single.queries[i])
        << "slice " << s << " dQ index " << i;
      ASSERT_EQ(batched_gradient_keys[s * slice + i], single.keys[i])
        << "slice " << s << " dK index " << i;
      ASSERT_EQ(batched_gradient_values[s * slice + i], single.values[i])
        << "slice " << s << " dV index " << i;
    }
  }
}

} // namespace Attention
} // namespace Transformer
} // namespace GoogleUnitTests
