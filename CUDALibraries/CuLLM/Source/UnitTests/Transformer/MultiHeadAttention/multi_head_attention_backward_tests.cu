#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/multi_head_attention.h"
#include "Transformer/MultiHeadAttention/multi_head_attention_backward.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::merge_qkv_heads;
using Transformer::MultiHeadAttention::multi_head_attention;
using Transformer::MultiHeadAttention::multi_head_attention_backward;
using Transformer::MultiHeadAttention::split_heads;
using Transformer::MultiHeadAttention::split_qkv_heads;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-0.5, 0.5] — half the
// amplitude of the forward tests' inputs, keeping the finite-difference
// probes of the gradient tests well away from softmax saturation.
//------------------------------------------------------------------------------
vector<float> make_backward_inputs(const int count, const int seed)
{
  vector<float> result(count);
  for (int i {0}; i < count; ++i)
  {
    result[i] =
      static_cast<float>((i * seed + 7 * (i % 13)) % 211 - 105) / 210.0f;
  }
  return result;
}

//------------------------------------------------------------------------------
// split_heads must be the exact inverse of merge_heads: both are
// permutations, so a round trip through merge then split must reproduce the
// per-head tensor bit-for-bit.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionBackwardTests, SplitHeadsInvertsMergeHeads)
{
  constexpr int kHD {32};
  constexpr int batch_size {2};
  constexpr int num_heads {3};
  constexpr int sequence_length {5};
  const int d_model {num_heads * kHD};
  const int per_head_elements {
    batch_size * num_heads * sequence_length * kHD};

  const vector<float> per_head {make_backward_inputs(per_head_elements, 3)};

  Array<float> d_per_head(per_head_elements);
  Array<float> d_concat(batch_size * sequence_length * d_model);
  Array<float> d_round_trip(per_head_elements);
  d_per_head.copy_host_input_to_device(per_head);

  constexpr int kThreadsPerBlock {256};
  const int number_of_blocks {
    (per_head_elements + kThreadsPerBlock - 1) / kThreadsPerBlock};

  ::Transformer::MultiHeadAttention::merge_heads<float, kHD>
    <<<number_of_blocks, kThreadsPerBlock>>>(
      d_concat.elements_,
      d_per_head.elements_,
      batch_size,
      num_heads,
      sequence_length);
  split_heads<float, kHD><<<number_of_blocks, kThreadsPerBlock>>>(
    d_round_trip.elements_,
    d_concat.elements_,
    batch_size,
    num_heads,
    sequence_length);
  cudaDeviceSynchronize();

  vector<float> round_trip(per_head_elements);
  d_round_trip.copy_device_output_to_host(round_trip);
  for (int i {0}; i < per_head_elements; ++i)
  {
    ASSERT_EQ(round_trip[i], per_head[i]) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// merge_qkv_heads must be the exact inverse of split_qkv_heads: a fused
// (B·T, 3·d_model) buffer split into Q/K/V and merged back must be
// reproduced bit-for-bit.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionBackwardTests, MergeQkvHeadsInvertsSplitQkvHeads)
{
  constexpr int kHD {32};
  constexpr int batch_size {2};
  constexpr int num_heads {3};
  constexpr int sequence_length {5};
  const int d_model {num_heads * kHD};
  const int qkv_elements {batch_size * sequence_length * 3 * d_model};
  const int per_head_elements {
    batch_size * num_heads * sequence_length * kHD};

  const vector<float> qkv {make_backward_inputs(qkv_elements, 7)};

  Array<float> d_qkv(qkv_elements);
  Array<float> d_queries(per_head_elements);
  Array<float> d_keys(per_head_elements);
  Array<float> d_values(per_head_elements);
  Array<float> d_round_trip(qkv_elements);
  d_qkv.copy_host_input_to_device(qkv);

  constexpr int kThreadsPerBlock {256};
  const int number_of_blocks {
    (per_head_elements + kThreadsPerBlock - 1) / kThreadsPerBlock};

  split_qkv_heads<float, kHD><<<number_of_blocks, kThreadsPerBlock>>>(
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    d_qkv.elements_,
    batch_size,
    num_heads,
    sequence_length);
  merge_qkv_heads<float, kHD><<<number_of_blocks, kThreadsPerBlock>>>(
    d_round_trip.elements_,
    d_queries.elements_,
    d_keys.elements_,
    d_values.elements_,
    batch_size,
    num_heads,
    sequence_length);
  cudaDeviceSynchronize();

  vector<float> round_trip(qkv_elements);
  d_round_trip.copy_device_output_to_host(round_trip);
  for (int i {0}; i < qkv_elements; ++i)
  {
    ASSERT_EQ(round_trip[i], qkv[i]) << "index " << i;
  }
}

//------------------------------------------------------------------------------
// Gradient-check fixture. Loss L := ⟨G, Y⟩ = Σ G ⊙ MHA(X) for a fixed
// pseudo-random G, so dL/dY = G is exactly the gradient_output fed to the
// backward pass, and every analytic parameter gradient can be checked
// against the central difference (L(θ+ε) − L(θ−ε)) / 2ε of the *device*
// forward pass — an end-to-end check of the entire backward chain
// (output linear map adjoint → head split → flash attention backward →
// QKV merge → QKV linear map adjoint) with no CPU re-derivation.
//------------------------------------------------------------------------------
template <int kHD, int kWarps, bool kCausal>
class GradientCheck
{
  public:

    static constexpr int batch_size {1};
    static constexpr int num_heads {2};
    static constexpr int sequence_length {8};
    static constexpr int d_model {num_heads * kHD};
    static constexpr int num_tokens {batch_size * sequence_length};
    static constexpr int per_head_elements {
      batch_size * num_heads * sequence_length * kHD};

    GradientCheck():
      input_{make_backward_inputs(num_tokens * d_model, 3)},
      qkv_weights_{make_backward_inputs(d_model * 3 * d_model, 5)},
      output_weights_{make_backward_inputs(d_model * d_model, 11)},
      loss_weights_{make_backward_inputs(num_tokens * d_model, 13)}
    {}

    // Device forward pass; returns L = ⟨G, Y⟩.
    double loss(
      const vector<float>& input,
      const vector<float>& qkv_weights,
      const vector<float>& output_weights)
    {
      Array<float> d_input(num_tokens * d_model);
      Array<float> d_qkv_weights(d_model * 3 * d_model);
      Array<float> d_output_weights(d_model * d_model);
      Array<float> d_qkv_workspace(num_tokens * 3 * d_model);
      Array<float> d_queries(per_head_elements);
      Array<float> d_keys(per_head_elements);
      Array<float> d_values(per_head_elements);
      Array<float> d_attention_output(per_head_elements);
      Array<float> d_concat_workspace(num_tokens * d_model);
      Array<float> d_output(num_tokens * d_model);

      d_input.copy_host_input_to_device(input);
      d_qkv_weights.copy_host_input_to_device(qkv_weights);
      d_output_weights.copy_host_input_to_device(output_weights);

      LibraryContextHandle handle {};
      Stream stream {};
      EXPECT_TRUE((multi_head_attention<float, kHD, kWarps, kCausal>(
        handle,
        stream,
        d_output.elements_,
        d_qkv_workspace.elements_,
        d_queries.elements_,
        d_keys.elements_,
        d_values.elements_,
        d_attention_output.elements_,
        d_concat_workspace.elements_,
        d_input.elements_,
        d_qkv_weights.elements_,
        d_output_weights.elements_,
        batch_size,
        num_heads,
        sequence_length)));
      cudaDeviceSynchronize();

      vector<float> output(num_tokens * d_model);
      d_output.copy_device_output_to_host(output);

      double total {0.0};
      for (size_t i {0}; i < output.size(); ++i)
      {
        total += static_cast<double>(loss_weights_[i]) *
          static_cast<double>(output[i]);
      }
      return total;
    }

    // Forward with saved activations, then the full backward pass;
    // fills the three analytic gradients.
    void analytic_gradients(
      vector<float>& gradient_input,
      vector<float>& gradient_qkv_weights,
      vector<float>& gradient_output_weights)
    {
      Array<float> d_input(num_tokens * d_model);
      Array<float> d_qkv_weights(d_model * 3 * d_model);
      Array<float> d_output_weights(d_model * d_model);
      Array<float> d_qkv_workspace(num_tokens * 3 * d_model);
      Array<float> d_queries(per_head_elements);
      Array<float> d_keys(per_head_elements);
      Array<float> d_values(per_head_elements);
      Array<float> d_attention_output(per_head_elements);
      Array<float> d_concat_workspace(num_tokens * d_model);
      Array<float> d_output(num_tokens * d_model);
      Array<float> d_logsumexp(
        batch_size * num_heads * sequence_length);

      d_input.copy_host_input_to_device(input_);
      d_qkv_weights.copy_host_input_to_device(qkv_weights_);
      d_output_weights.copy_host_input_to_device(output_weights_);

      LibraryContextHandle handle {};
      Stream stream {};
      ASSERT_TRUE((multi_head_attention<float, kHD, kWarps, kCausal>(
        handle,
        stream,
        d_output.elements_,
        d_qkv_workspace.elements_,
        d_queries.elements_,
        d_keys.elements_,
        d_values.elements_,
        d_attention_output.elements_,
        d_concat_workspace.elements_,
        d_input.elements_,
        d_qkv_weights.elements_,
        d_output_weights.elements_,
        batch_size,
        num_heads,
        sequence_length,
        d_logsumexp.elements_)));
      cudaDeviceSynchronize();

      Array<float> d_gradient_output(num_tokens * d_model);
      Array<float> d_gradient_input(num_tokens * d_model);
      Array<float> d_gradient_qkv_weights(d_model * 3 * d_model);
      Array<float> d_gradient_output_weights(d_model * d_model);
      Array<float> d_gradient_concat(num_tokens * d_model);
      Array<float> d_gradient_attention_output(per_head_elements);
      Array<float> d_gradient_queries(per_head_elements);
      Array<float> d_gradient_keys(per_head_elements);
      Array<float> d_gradient_values(per_head_elements);
      Array<float> d_gradient_qkv(num_tokens * 3 * d_model);
      Array<float> d_row_dots(batch_size * num_heads * sequence_length);

      d_gradient_output.copy_host_input_to_device(loss_weights_);

      ASSERT_TRUE((
        multi_head_attention_backward<float, kHD, kWarps, kCausal>(
          handle,
          stream,
          d_gradient_input.elements_,
          d_gradient_qkv_weights.elements_,
          d_gradient_output_weights.elements_,
          d_concat_workspace.elements_,
          d_gradient_concat.elements_,
          d_gradient_attention_output.elements_,
          d_gradient_queries.elements_,
          d_gradient_keys.elements_,
          d_gradient_values.elements_,
          d_gradient_qkv.elements_,
          d_row_dots.elements_,
          d_gradient_output.elements_,
          d_input.elements_,
          d_qkv_weights.elements_,
          d_output_weights.elements_,
          d_queries.elements_,
          d_keys.elements_,
          d_values.elements_,
          d_attention_output.elements_,
          d_logsumexp.elements_,
          batch_size,
          num_heads,
          sequence_length)));
      cudaDeviceSynchronize();

      gradient_input.resize(num_tokens * d_model);
      gradient_qkv_weights.resize(d_model * 3 * d_model);
      gradient_output_weights.resize(d_model * d_model);
      d_gradient_input.copy_device_output_to_host(gradient_input);
      d_gradient_qkv_weights.copy_device_output_to_host(gradient_qkv_weights);
      d_gradient_output_weights.copy_device_output_to_host(
        gradient_output_weights);
    }

    enum class Tensor { kInput, kQkvWeights, kOutputWeights };

    // Central difference dL/dθ_i for one entry of the selected parameter
    // tensor: perturbs a fresh copy of that tensor around its nominal value
    // and reruns the device forward pass twice.
    double finite_difference(const Tensor which, const size_t index)
    {
      // eps large enough that the float forward's noise (~1e-6 relative)
      // stays well below the difference quotient.
      const float eps {1e-2f};
      vector<float> input {input_};
      vector<float> qkv_weights {qkv_weights_};
      vector<float> output_weights {output_weights_};
      vector<float>& target {
        which == Tensor::kInput ? input :
        which == Tensor::kQkvWeights ? qkv_weights : output_weights};

      const float saved {target[index]};
      target[index] = saved + eps;
      const double loss_plus {loss(input, qkv_weights, output_weights)};
      target[index] = saved - eps;
      const double loss_minus {loss(input, qkv_weights, output_weights)};
      return (loss_plus - loss_minus) / (2.0 * static_cast<double>(eps));
    }

    void check_all(const int probes_per_tensor)
    {
      vector<float> gradient_input;
      vector<float> gradient_qkv_weights;
      vector<float> gradient_output_weights;
      analytic_gradients(
        gradient_input, gradient_qkv_weights, gradient_output_weights);

      struct Target
      {
        Tensor which;
        const vector<float>* analytic;
        const char* name;
      };
      const Target targets[3] {
        {Tensor::kInput, &gradient_input, "input"},
        {Tensor::kQkvWeights, &gradient_qkv_weights, "qkv_weights"},
        {Tensor::kOutputWeights, &gradient_output_weights, "output_weights"}};

      for (const Target& target : targets)
      {
        const size_t size {target.analytic->size()};
        const size_t stride {size / probes_per_tensor + 1};
        for (size_t index {0}; index < size; index += stride)
        {
          const double numeric {finite_difference(target.which, index)};
          const double analytic {
            static_cast<double>((*target.analytic)[index])};
          // Float forward + O(ε²) truncation: 2e-2 absolute floor, 5%
          // relative.
          const double tolerance {2e-2 + 5e-2 * std::abs(numeric)};
          EXPECT_NEAR(analytic, numeric, tolerance)
            << target.name << " index " << index;
        }
      }
    }

    vector<float> input_;
    vector<float> qkv_weights_;
    vector<float> output_weights_;
    vector<float> loss_weights_;
};

//------------------------------------------------------------------------------
// Every analytic gradient (dX, dW_qkv, dW^O) against central differences of
// the device forward pass.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionBackwardTests, GradientsMatchFiniteDifferences)
{
  GradientCheck<32, 4, false> check {};
  check.check_all(12);
}

//------------------------------------------------------------------------------
// Causal variant: the backward recomputes weights under the same mask as
// the forward (kCausal threads through both), so the same finite-difference
// check must hold.
//------------------------------------------------------------------------------
TEST(MultiHeadAttentionBackwardTests, CausalGradientsMatchFiniteDifferences)
{
  GradientCheck<32, 4, true> check {};
  check.check_all(12);
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
