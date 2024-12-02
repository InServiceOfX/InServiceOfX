#include "Drafts/LLM/AttentionForward/softmax.h"

#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::exp;
using std::vector;

namespace GoogleUnitTests
{
namespace Drafts
{
namespace LLM
{
namespace AttentionForward
{

using ::Drafts::LLM::AttentionForward::attention_softmax_kernel1;
using ::Drafts::LLM::AttentionForward::softmax_forward_kernel4;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(AttentionSoftmaxKernel1Tests, AttentionSoftMaxKernel1Computes)
{
  const int batch_size {1};
  const int seq_length {4};
  const int num_heads {1};

  // Test case: [3.0, 1.0, 2.0, 4.0]
  vector<float> input_data(batch_size * num_heads * seq_length * seq_length, 0.0f);
  // First row (t=0)
  input_data[0] = 3.0f;
  // Second row (t=1)
  input_data[seq_length + 0] = 3.0f;
  input_data[seq_length + 1] = 1.0f;
  // Third row (t=2)
  input_data[2 * seq_length + 0] = 3.0f;
  input_data[2 * seq_length + 1] = 1.0f;
  input_data[2 * seq_length + 2] = 2.0f;

  Array<float> d_input(input_data.size());
  Array<float> d_output(input_data.size());
  d_input.copy_host_input_to_device(input_data);

  const dim3 block_size(256);
  const dim3 grid_size(
    (batch_size * num_heads * seq_length + block_size.x - 1) / block_size.x);

  attention_softmax_kernel1<<<grid_size, block_size>>>(
    d_output.elements_,
    d_input.elements_,
    batch_size,
    seq_length,
    num_heads);

  vector<float> output(input_data.size());
  d_output.copy_device_output_to_host(output);

  // For t=0: maxval=3.0
  // Only x[0] processed, rest masked
  EXPECT_NEAR(output[0], 1.0f, 1e-6f);
  EXPECT_NEAR(output[1], 0.0f, 1e-6f);
  EXPECT_NEAR(output[2], 0.0f, 1e-6f);
  EXPECT_NEAR(output[3], 0.0f, 1e-6f);

  // For t=1: maxval=3.0
  const int idx1 {seq_length};
  const float exp0_t1 {exp(3.0f - 3.0f)};  // 1.0
  const float exp1_t1 {exp(1.0f - 3.0f)};  // exp(-2)
  const float sum_t1 {exp0_t1 + exp1_t1};
  EXPECT_NEAR(output[idx1 + 0], exp0_t1/sum_t1, 1e-6f);
  EXPECT_NEAR(output[idx1 + 1], exp1_t1/sum_t1, 1e-6f);
  EXPECT_NEAR(output[idx1 + 2], 0.0f, 1e-6f);
  EXPECT_NEAR(output[idx1 + 3], 0.0f, 1e-6f);

  // For t=2: maxval=3.0
  const int idx2 {2 * seq_length};
  const float exp0_t2 {exp(3.0f - 3.0f)};  // 1.0
  const float exp1_t2 {exp(1.0f - 3.0f)};  // exp(-2)
  const float exp2_t2 {exp(2.0f - 3.0f)};  // exp(-1)
  const float sum_t2 {exp0_t2 + exp1_t2 + exp2_t2};
  EXPECT_NEAR(output[idx2 + 0], exp0_t2/sum_t2, 1e-6f);
  EXPECT_NEAR(output[idx2 + 1], exp1_t2/sum_t2, 1e-6f);
  EXPECT_NEAR(output[idx2 + 2], exp2_t2/sum_t2, 1e-6f);
  EXPECT_NEAR(output[idx2 + 3], 0.0f, 1e-6f);
}

TEST(SoftmaxForwardKernel4Tests, SingleRowBasic)
{
  // Test single row with simple values
  const int N {1};  // One row
  const int C {4};  // Four elements
  
  vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  vector<float> output(N * C);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  // Use 32 threads (one warp) for this simple test
  const int block_size {32};
  const size_t shared_mem_size {2 * (block_size / 32) * sizeof(float)};
  
  softmax_forward_kernel4<<<1, block_size, shared_mem_size>>>(
    d_output.elements_,
    d_input.elements_,
    N,
    C);
  
  d_output.copy_device_output_to_host(output);
  
  // Compute expected softmax values.
  // Let M_x = \max_{i \in 0,.. N * C -1} x_i, then
  // y_i = exp{(x_i - M_x)} / \sum_{i=0}^{N * C - 1} exp{(x_i - M_x)}
  const float max_val {4.0f};
  vector<float> expected(N * C);
  float sum {0.0f};
  for (int i {0}; i < C; ++i)
  {
    expected[i] = std::exp(input[i] - max_val);
    sum += expected[i];
  }
  for (int i {0}; i < C; ++i)
  {
    expected[i] /= sum;
  }
  
  // Verify results
  for (int i {0}; i < C; ++i)
  {
    EXPECT_NEAR(output[i], expected[i], 1e-3f);
  }
}

TEST(SoftmaxForwardKernel4Tests, NegativeValues)
{
  const int N {1};  // One row
  const int C {4};  // Four elements
  
  // Test with negative values and zero
  vector<float> input = {-3.0f, -2.0f, -1.0f, 0.0f};
  vector<float> output(N * C);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  const int block_size {32};
  const size_t shared_mem_size {2 * (block_size / 32) * sizeof(float)};
  
  softmax_forward_kernel4<<<1, block_size, shared_mem_size>>>(
    d_output.elements_,
    d_input.elements_,
    N,
    C);
  
  d_output.copy_device_output_to_host(output);
  
  // Compute expected softmax values
  // max_val is 0.0f from the input array
  const float max_val {0.0f};
  vector<float> expected(N * C);
  float sum {0.0f};
  for (int i {0}; i < C; ++i)
  {
    expected[i] = std::exp(input[i] - max_val);
    sum += expected[i];
  }
  for (int i {0}; i < C; ++i)
  {
    expected[i] /= sum;
  }
  
  // Verify results
  for (int i {0}; i < C; ++i)
  {
    EXPECT_NEAR(output[i], expected[i], 1e-3f);
  }
}

TEST(SoftmaxForwardKernel4Tests, TwoRows)
{
  const unsigned int N {2};  // Two rows
  const unsigned int C {4};  // Four elements per row
  
  // Test with different values for each row
  vector<float> input = {
    1.0f, 2.0f, 3.0f, 4.0f,     // First row
    -3.0f, -2.0f, -1.0f, 0.0f   // Second row
  };
  vector<float> output(N * C);
  
  Array<float> d_input(input.size());
  Array<float> d_output(output.size());
  d_input.copy_host_input_to_device(input);
  
  const unsigned int block_size {32};
  const unsigned int shared_mem_size {2 * (block_size / 32) * sizeof(float)};
  
  // Launch N blocks, one for each row
  softmax_forward_kernel4<<<N, block_size, shared_mem_size>>>(
    d_output.elements_,
    d_input.elements_,
    N,
    C);
  
  d_output.copy_device_output_to_host(output);
  
  // Compute expected softmax values for each row independently
  vector<float> expected(N * C);
  
  // First row: max_val = 4.0f
  float sum1 {0.0f};
  for (int i {0}; i < C; ++i)
  {
    expected[i] = std::exp(input[i] - 4.0f);
    sum1 += expected[i];
  }
  for (int i {0}; i < C; ++i)
  {
    expected[i] /= sum1;
  }
  
  // Second row: max_val = 0.0f
  float sum2 {0.0f};
  for (int i {0}; i < C; ++i)
  {
    expected[C + i] = std::exp(input[C + i] - 0.0f);
    sum2 += expected[C + i];
  }
  for (int i {0}; i < C; ++i)
  {
    expected[C + i] /= sum2;
  }
  
  // Verify results for both rows
  for (int n {0}; n < N; ++n)
  {
    for (int i {0}; i < C; ++i)
    {
      EXPECT_NEAR(output[n * C + i], expected[n * C + i], 1e-3f);
    }
  }
}

} // namespace AttentionForward
} // namespace LLM
} // namespace Drafts
} // namespace GoogleUnitTests
