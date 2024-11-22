#include "LLM/attention_forward.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"

#include <vector>
#include <cmath>
#include <random>

using DataStructures::Array;
using std::vector;
using LLM::attention_query_key_kernel1;

namespace GoogleUnitTests
{
namespace LLM
{

class AttentionForwardTests : public ::testing::Test
{
protected:
    // Helper function to compute expected scaled dot product
    float compute_expected_dot_product(
        const vector<float>& query,
        const vector<float>& key,
        int head_size)
    {
        float dot_product = 0.0f;
        for (int i = 0; i < head_size; ++i)
        {
            dot_product += query[i] * key[i];
        }
        return dot_product / std::sqrt(static_cast<float>(head_size));
    }
};

// Test basic functionality with small dimensions
TEST_F(AttentionForwardTests, BasicFunctionality)
{
    const int batch_size = 1;
    const int seq_length = 4;
    const int hidden_size = 8;
    const int num_heads = 2;

    vector<float> input_data(batch_size * seq_length * hidden_size * 3, 0.0f);
    
    // Set simple test case: first query and key vectors are [1,1,0,0]
    for (int i {0}; i < 2; ++i)
    {
        input_data[i] = 1.0f;                    // Query
        input_data[hidden_size + i] = 1.0f;      // Key
    }

    Array<float> d_input(input_data.size());
    Array<float> d_output(batch_size * num_heads * seq_length * seq_length);

    d_input.copy_host_input_to_device(input_data);

    dim3 block_size(256);
    dim3 grid_size((batch_size * num_heads * seq_length * seq_length + 
        block_size.x - 1) / block_size.x);

    attention_query_key_kernel1<<<grid_size, block_size>>>(
        d_output.elements_,
        d_input.elements_,
        batch_size,
        seq_length,
        hidden_size,
        num_heads);

    vector<float> output(batch_size * num_heads * seq_length * seq_length);
    d_output.copy_device_output_to_host(output);

    // Expected: (1*1 + 1*1) / sqrt(4) = 1.0
    EXPECT_NEAR(output[0], 1.0f, 1e-6);
    
    // Check autoregressive mask
    for (int t = 0; t < seq_length; ++t)
    {
        for (int t2 = t + 1; t2 < seq_length; ++t2)
        {
            EXPECT_EQ(output[t * seq_length + t2], 
                -std::numeric_limits<float>::infinity());
        }
    }
}

// Test edge case with maximum sequence length
TEST_F(AttentionForwardTests, MaxSequenceLength)
{
    const int batch_size = 1;
    const int seq_length = 2048;  // Common max sequence length
    const int hidden_size = 8;
    const int num_heads = 2;

    vector<float> input_data(batch_size * seq_length * hidden_size * 3, 0.0f);
    Array<float> d_input(input_data.size());
    Array<float> d_output(batch_size * num_heads * seq_length * seq_length);

    d_input.copy_host_input_to_device(input_data);

    dim3 block_size(256);
    dim3 grid_size((batch_size * num_heads * seq_length * seq_length + 
        block_size.x - 1) / block_size.x);

    attention_query_key_kernel1<<<grid_size, block_size>>>(
        d_output.elements_,
        d_input.elements_,
        batch_size,
        seq_length,
        hidden_size,
        num_heads);

    vector<float> output(batch_size * num_heads * seq_length * seq_length);
    d_output.copy_device_output_to_host(output);

    // Verify first position (should be 0 due to zero initialization)
    EXPECT_NEAR(output[0], 0.0f, 1e-6);
}

// Test mathematical properties with random data
TEST_F(AttentionForwardTests, MathematicalProperties)
{
    const int batch_size = 1;
    const int seq_length = 4;
    const int hidden_size = 8;
    const int num_heads = 2;
    const int head_size = hidden_size / num_heads;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    vector<float> input_data(batch_size * seq_length * hidden_size * 3);
    for (auto& val : input_data)
    {
        val = dist(gen);
    }

    Array<float> d_input(input_data.size());
    Array<float> d_output(batch_size * num_heads * seq_length * seq_length);

    d_input.copy_host_input_to_device(input_data);

    dim3 block_size(256);
    dim3 grid_size((batch_size * num_heads * seq_length * seq_length + 
        block_size.x - 1) / block_size.x);

    attention_query_key_kernel1<<<grid_size, block_size>>>(
        d_output.elements_,
        d_input.elements_,
        batch_size,
        seq_length,
        hidden_size,
        num_heads);

    vector<float> output(batch_size * num_heads * seq_length * seq_length);
    d_output.copy_device_output_to_host(output);

    // Test symmetry property for first head
    for (int t1 = 0; t1 < seq_length; ++t1)
    {
        for (int t2 = 0; t2 <= t1; ++t2)
        {
            vector<float> query(head_size);
            vector<float> key(head_size);
            
            // Extract query and key vectors
            for (int i {0}; i < head_size; ++i)
            {
                query[i] = input_data[t1 * hidden_size * 3 + i];
                key[i] = input_data[t2 * hidden_size * 3 + hidden_size + i];
            }

            float expected = compute_expected_dot_product(query, key, head_size);
            EXPECT_NEAR(output[t1 * seq_length + t2], expected, 1e-5);
        }
    }
}

// Test multiple batch and head processing
TEST_F(AttentionForwardTests, MultipleBatchAndHeads)
{
    const int batch_size = 2;
    const int seq_length = 4;
    const int hidden_size = 8;
    const int num_heads = 2;

    vector<float> input_data(batch_size * seq_length * hidden_size * 3, 1.0f);
    Array<float> d_input(input_data.size());
    Array<float> d_output(batch_size * num_heads * seq_length * seq_length);

    d_input.copy_host_input_to_device(input_data);

    // Test different thread block configurations
    std::vector<dim3> block_sizes = {{128}, {256}, {512}};
    
    for (const auto& block_size : block_sizes)
    {
        dim3 grid_size((batch_size * num_heads * seq_length * seq_length + 
            block_size.x - 1) / block_size.x);

        attention_query_key_kernel1<<<grid_size, block_size>>>(
            d_output.elements_,
            d_input.elements_,
            batch_size,
            seq_length,
            hidden_size,
            num_heads);

        vector<float> output(batch_size * num_heads * seq_length * seq_length);
        d_output.copy_device_output_to_host(output);

        // Verify results are consistent across different block sizes
        float expected = hidden_size / (2.0f * std::sqrt(hidden_size / 2.0f));
        for (int b = 0; b < batch_size; ++b)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                for (int t1 = 0; t1 < seq_length; ++t1)
                {
                    for (int t2 = 0; t2 <= t1; ++t2)
                    {
                        int idx = b * (num_heads * seq_length * seq_length) +
                                h * (seq_length * seq_length) +
                                t1 * seq_length + t2;
                        EXPECT_NEAR(output[idx], expected, 1e-5);
                    }
                }
            }
        }
    }
}

} // namespace LLM
} // namespace GoogleUnitTests
