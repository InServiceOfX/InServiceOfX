#include "LLM/AttentionForward/softmax.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"
#include <vector>

using DataStructures::Array;
using LLM::AttentionForward::softmax_forward_kernel4;
using std::vector;

namespace GoogleUnitTests
{
namespace LLM
{
namespace AttentionForward
{

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
    EXPECT_NEAR(output[i], expected[i], 1e-6f);
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
    EXPECT_NEAR(output[i], expected[i], 1e-6f);
  }
}

} // namespace AttentionForward
} // namespace LLM
} // namespace GoogleUnitTests