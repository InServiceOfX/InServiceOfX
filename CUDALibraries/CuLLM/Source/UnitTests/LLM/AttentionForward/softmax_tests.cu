#include "LLM/AttentionForward/softmax.h"

#include "DataStructures/Array.h"
#include "gtest/gtest.h"
#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;
using LLM::AttentionForward::softmax_forward;

namespace GoogleUnitTests
{

namespace LLM
{
namespace AttentionForward
{

TEST(SoftmaxForwardTests, SingleRowBasic)
{
  const unsigned int N {1};  // One batch
  const unsigned int T {4};  // Sequence length
  
  // Initialize full attention matrix (T*T) with input values
  vector<float> input(N * T * T, 0.0f);
  // First row (t=0): only position 0
  input[0] = 1.0f;
  // Second row (t=1): positions 0,1
  input[T + 0] = 1.0f;
  input[T + 1] = 2.0f;
  // Third row (t=2): positions 0,1,2
  input[2 * T + 0] = 1.0f;
  input[2 * T + 1] = 2.0f;
  input[2 * T + 2] = 3.0f;
  // Fourth row (t=3): all positions
  input[3 * T + 0] = 1.0f;
  input[3 * T + 1] = 2.0f;
  input[3 * T + 2] = 3.0f;
  input[3 * T + 3] = 4.0f;
  
  vector<float> output(N * T * T);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  const float inverse_temperature {1.0f};
  const unsigned int block_size {32};
  
  softmax_forward<float><<<N * T, block_size>>>(
    d_output.elements_,
    inverse_temperature,
    d_input.elements_,
    N,
    T);
  
  d_output.copy_device_output_to_host(output);
  
  // Position 0: Only sees itself
  EXPECT_NEAR(output[0], 1.0f, 1e-3f);
  EXPECT_NEAR(output[1], 0.0f, 1e-3f);
  EXPECT_NEAR(output[2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[3], 0.0f, 1e-3f);
  
  // Position 1: Sees positions 0 and 1
  const float max_val1 {2.0f};
  const float exp0_1 {exp(1.0f - max_val1)};
  const float exp1_1 {exp(2.0f - max_val1)};
  const float sum_1 {exp0_1 + exp1_1};
  EXPECT_NEAR(output[T + 0], exp0_1/sum_1, 1e-3f);
  EXPECT_NEAR(output[T + 1], exp1_1/sum_1, 1e-3f);
  EXPECT_NEAR(output[T + 2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[T + 3], 0.0f, 1e-3f);
  
  // Position 2: Sees positions 0, 1, and 2
  const float max_val2 {3.0f};
  const float exp0_2 {exp(1.0f - max_val2)};
  const float exp1_2 {exp(2.0f - max_val2)};
  const float exp2_2 {exp(3.0f - max_val2)};
  const float sum_2 {exp0_2 + exp1_2 + exp2_2};
  EXPECT_NEAR(output[2 * T + 0], exp0_2/sum_2, 1e-3f);
  EXPECT_NEAR(output[2 * T + 1], exp1_2/sum_2, 1e-3f);
  EXPECT_NEAR(output[2 * T + 2], exp2_2/sum_2, 1e-3f);
  EXPECT_NEAR(output[2 * T + 3], 0.0f, 1e-3f);
  
  // Position 3: Sees all positions
  const float max_val3 {4.0f};
  const float exp0_3 {exp(1.0f - max_val3)};
  const float exp1_3 {exp(2.0f - max_val3)};
  const float exp2_3 {exp(3.0f - max_val3)};
  const float exp3_3 {exp(4.0f - max_val3)};
  const float sum_3 {exp0_3 + exp1_3 + exp2_3 + exp3_3};
  EXPECT_NEAR(output[3 * T + 0], exp0_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[3 * T + 1], exp1_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[3 * T + 2], exp2_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[3 * T + 3], exp3_3/sum_3, 1e-3f);
}

TEST(SoftmaxForwardTests, TemperatureScaling)
{
  const unsigned int N {1};
  const unsigned int T {4};
  
  vector<float> input(N * T * T, 0.0f);
  input[0] = 1.0f;
  input[T + 0] = 1.0f;
  input[T + 1] = 2.0f;
  input[2 * T + 0] = 1.0f;
  input[2 * T + 1] = 2.0f;
  input[2 * T + 2] = 3.0f;
  input[3 * T + 0] = 1.0f;
  input[3 * T + 1] = 2.0f;
  input[3 * T + 2] = 3.0f;
  input[3 * T + 3] = 4.0f;
  vector<float> output(N * T * T);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  // Higher temperature = softer distribution
  const float inv_temperature {0.5f};
  const unsigned int block_size {32};
  
  softmax_forward<float><<<N * T, block_size>>>(
    d_output.elements_,
    inv_temperature,
    d_input.elements_,
    N,
    T);
  
  d_output.copy_device_output_to_host(output);
  
  // With temperature scaling, exponentials are less extreme
  const float exp0_1 {exp(0.5f * (1.0f - 2.0f))};
  const float exp1_1 {exp(0.5f * (2.0f - 2.0f))};
  const float sum_1 {exp0_1 + exp1_1};
  EXPECT_NEAR(output[T + 0], exp0_1/sum_1, 1e-3f);
  EXPECT_NEAR(output[T + 1], exp1_1/sum_1, 1e-3f);
}

TEST(SoftmaxForwardTests, TwoBatches)
{
  const unsigned int N {2};  // Two batches
  const unsigned int T {4};  // Sequence length
  
  // Initialize attention matrices for both batches
  vector<float> input(N * T * T, 0.0f);
  
  // First batch
  // Row 0
  input[0] = 1.0f;
  // Row 1
  input[T + 0] = 1.0f;
  input[T + 1] = 2.0f;
  // Row 2
  input[2 * T + 0] = 1.0f;
  input[2 * T + 1] = 2.0f;
  input[2 * T + 2] = 3.0f;
  // Row 3
  input[3 * T + 0] = 1.0f;
  input[3 * T + 1] = 2.0f;
  input[3 * T + 2] = 3.0f;
  input[3 * T + 3] = 4.0f;
  
  // Second batch (different values)
  const unsigned int batch2_offset {T * T};
  // Row 0
  input[batch2_offset + 0] = 0.5f;
  // Row 1
  input[batch2_offset + T + 0] = 0.5f;
  input[batch2_offset + T + 1] = 1.5f;
  // Row 2
  input[batch2_offset + 2 * T + 0] = 0.5f;
  input[batch2_offset + 2 * T + 1] = 1.5f;
  input[batch2_offset + 2 * T + 2] = 2.5f;
  // Row 3
  input[batch2_offset + 3 * T + 0] = 0.5f;
  input[batch2_offset + 3 * T + 1] = 1.5f;
  input[batch2_offset + 3 * T + 2] = 2.5f;
  input[batch2_offset + 3 * T + 3] = 3.5f;
  
  vector<float> output(N * T * T);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  const float inverse_temperature {1.0f};
  const unsigned int block_size {32};
  
  softmax_forward<float><<<N * T, block_size>>>(
    d_output.elements_,
    inverse_temperature,
    d_input.elements_,
    N,
    T);
  
  d_output.copy_device_output_to_host(output);
  
  // Test first batch (same as SingleRowBasic test)
  // Position 0
  EXPECT_NEAR(output[0], 1.0f, 1e-3f);
  EXPECT_NEAR(output[1], 0.0f, 1e-3f);
  EXPECT_NEAR(output[2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[3], 0.0f, 1e-3f);
  
  // Position 1
  const float max_val1 {2.0f};
  const float exp0_1 {exp(1.0f - max_val1)};
  const float exp1_1 {exp(2.0f - max_val1)};
  const float sum_1 {exp0_1 + exp1_1};
  EXPECT_NEAR(output[T + 0], exp0_1/sum_1, 1e-3f);
  EXPECT_NEAR(output[T + 1], exp1_1/sum_1, 1e-3f);
  EXPECT_NEAR(output[T + 2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[T + 3], 0.0f, 1e-3f);
  
  // Test second batch
  // Position 0
  EXPECT_NEAR(output[batch2_offset + 0], 1.0f, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 1], 0.0f, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 3], 0.0f, 1e-3f);
  
  // Position 1
  const float max_val1_b2 {1.5f};
  const float exp0_1_b2 {exp(0.5f - max_val1_b2)};
  const float exp1_1_b2 {exp(1.5f - max_val1_b2)};
  const float sum_1_b2 {exp0_1_b2 + exp1_1_b2};
  EXPECT_NEAR(output[batch2_offset + T + 0], exp0_1_b2/sum_1_b2, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + T + 1], exp1_1_b2/sum_1_b2, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + T + 2], 0.0f, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + T + 3], 0.0f, 1e-3f);

  const float max_val3 {3.5f};
  const float exp0_3 {exp(0.5f - max_val3)};
  const float exp1_3 {exp(1.5f - max_val3)};
  const float exp2_3 {exp(2.5f - max_val3)};
  const float exp3_3 {exp(3.5f - max_val3)};
  const float sum_3 {exp0_3 + exp1_3 + exp2_3 + exp3_3};
  EXPECT_NEAR(output[batch2_offset + 3 * T + 0], exp0_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 3 * T + 1], exp1_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 3 * T + 2], exp2_3/sum_3, 1e-3f);
  EXPECT_NEAR(output[batch2_offset + 3 * T + 3], exp3_3/sum_3, 1e-3f);
}

} // namespace AttentionForward
} // namespace LLM
} // namespace GoogleUnitTests