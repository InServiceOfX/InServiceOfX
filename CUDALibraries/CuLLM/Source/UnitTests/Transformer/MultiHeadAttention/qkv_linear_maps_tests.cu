#include "cuBLASWrappers/LibraryContextHandle.h"
#include "DataStructures/Array.h"
#include "StreamManagement/Stream.h"
#include "Transformer/MultiHeadAttention/qkv_linear_maps.h"
#include "gtest/gtest.h"

#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;
using Transformer::MultiHeadAttention::qkv_linear_maps;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_linear_map_inputs(const int count, const int seed)
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
// CPU reference: qkv = X @ W_qkv (row-major, double-precision accumulation),
// then the exact split_qkv_heads gather.
//------------------------------------------------------------------------------
void qkv_linear_maps_cpu(
  vector<float>& queries,
  vector<float>& keys,
  vector<float>& values,
  const vector<float>& input,
  const vector<float>& qkv_weight_matrix,
  const int batch_size,
  const int num_heads,
  const int head_dim,
  const int sequence_length)
{
  const int d_model {num_heads * head_dim};
  const int num_tokens {batch_size * sequence_length};

  vector<double> qkv(static_cast<size_t>(num_tokens) * 3 * d_model, 0.0);
  for (int row {0}; row < num_tokens; ++row)
  {
    for (int col {0}; col < 3 * d_model; ++col)
    {
      double accumulated {0.0};
      for (int k {0}; k < d_model; ++k)
      {
        accumulated += static_cast<double>(input[row * d_model + k]) *
          static_cast<double>(qkv_weight_matrix[k * 3 * d_model + col]);
      }
      qkv[static_cast<size_t>(row) * 3 * d_model + col] = accumulated;
    }
  }

  queries.assign(static_cast<size_t>(batch_size) * num_heads *
    sequence_length * head_dim, 0.0f);
  keys.assign(queries.size(), 0.0f);
  values.assign(queries.size(), 0.0f);

  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
      for (int t {0}; t < sequence_length; ++t)
      {
        const int qkv_row {b * sequence_length + t};
        const size_t out_index {
          static_cast<size_t>((b * num_heads + h) * sequence_length + t) *
            head_dim};
        for (int d {0}; d < head_dim; ++d)
        {
          queries[out_index + d] = static_cast<float>(
            qkv[static_cast<size_t>(qkv_row) * 3 * d_model + h * head_dim +
              d]);
          keys[out_index + d] = static_cast<float>(
            qkv[static_cast<size_t>(qkv_row) * 3 * d_model + d_model +
              h * head_dim + d]);
          values[out_index + d] = static_cast<float>(
            qkv[static_cast<size_t>(qkv_row) * 3 * d_model + 2 * d_model +
              h * head_dim + d]);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Full GEMM + split against the CPU reference, sizes chosen so d_model,
// num_tokens, and 3*d_model are all non-trivial (exercises real cuBLASLt
// tiling, not a toy single-tile case).
//------------------------------------------------------------------------------
TEST(QkvLinearMapsTests, MatchesCpuReference)
{
  constexpr int kHD {16};
  constexpr int batch_size {2};
  constexpr int num_heads {4};
  constexpr int sequence_length {20};
  constexpr int d_model {num_heads * kHD};
  const int num_tokens {batch_size * sequence_length};

  const vector<float> input {
    make_linear_map_inputs(num_tokens * d_model, 3)};
  const vector<float> qkv_weight_matrix {
    make_linear_map_inputs(d_model * 3 * d_model, 5)};

  Array<float> d_input(num_tokens * d_model);
  Array<float> d_weights(d_model * 3 * d_model);
  Array<float> d_workspace(num_tokens * 3 * d_model);
  Array<float> d_q(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_k(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_v(batch_size * num_heads * sequence_length * kHD);
  d_input.copy_host_input_to_device(input);
  d_weights.copy_host_input_to_device(qkv_weight_matrix);

  LibraryContextHandle handle {};
  Stream stream {};

  ASSERT_TRUE((qkv_linear_maps<float, kHD>(
    handle,
    stream,
    d_q.elements_,
    d_k.elements_,
    d_v.elements_,
    d_workspace.elements_,
    d_input.elements_,
    d_weights.elements_,
    batch_size,
    num_heads,
    sequence_length)));
  cudaDeviceSynchronize();

  vector<float> q(d_q.number_of_elements_);
  vector<float> k(d_k.number_of_elements_);
  vector<float> v(d_v.number_of_elements_);
  d_q.copy_device_output_to_host(q);
  d_k.copy_device_output_to_host(k);
  d_v.copy_device_output_to_host(v);

  vector<float> expected_q;
  vector<float> expected_k;
  vector<float> expected_v;
  qkv_linear_maps_cpu(
    expected_q, expected_k, expected_v,
    input, qkv_weight_matrix, batch_size, num_heads, kHD, sequence_length);

  ASSERT_EQ(q.size(), expected_q.size());
  for (size_t i {0}; i < q.size(); ++i)
  {
    ASSERT_NEAR(q[i], expected_q[i], 1e-3f) << "Q index " << i;
    ASSERT_NEAR(k[i], expected_k[i], 1e-3f) << "K index " << i;
    ASSERT_NEAR(v[i], expected_v[i], 1e-3f) << "V index " << i;
  }
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
