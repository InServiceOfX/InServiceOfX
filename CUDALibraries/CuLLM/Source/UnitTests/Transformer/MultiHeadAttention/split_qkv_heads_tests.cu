#include "DataStructures/Array.h"
#include "Transformer/MultiHeadAttention/split_qkv_heads.h"
#include "gtest/gtest.h"

#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::MultiHeadAttention::merge_heads;
using Transformer::MultiHeadAttention::split_qkv_heads;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic values so each element's expected destination is easy to
// hand-verify: qkv[row, col] = row * 1000 + col. Every gathered value is
// then traceable back to exactly which (row, col) it came from.
//------------------------------------------------------------------------------
vector<float> make_indexed_qkv(const int num_tokens, const int width)
{
  vector<float> result(num_tokens * width);
  for (int row {0}; row < num_tokens; ++row)
  {
    for (int col {0}; col < width; ++col)
    {
      result[row * width + col] =
        static_cast<float>(row) * 1000.0f + static_cast<float>(col);
    }
  }
  return result;
}

//------------------------------------------------------------------------------
// B=1, NH=1: the split degenerates to slicing three contiguous kHeadDim
// blocks out of each row. Hand-checkable against make_indexed_qkv's values.
//------------------------------------------------------------------------------
TEST(SplitQkvHeadsTests, SingleBatchSingleHeadSlicesContiguousBlocks)
{
  constexpr int kHD {4};
  constexpr int batch_size {1};
  constexpr int num_heads {1};
  constexpr int sequence_length {3};
  constexpr int d_model {num_heads * kHD};

  const vector<float> qkv {
    make_indexed_qkv(batch_size * sequence_length, 3 * d_model)};

  Array<float> d_qkv(qkv.size());
  Array<float> d_q(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_k(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_v(batch_size * num_heads * sequence_length * kHD);
  d_qkv.copy_host_input_to_device(qkv);

  split_qkv_heads<float, kHD><<<4, 32>>>(
    d_q.elements_, d_k.elements_, d_v.elements_, d_qkv.elements_,
    batch_size, num_heads, sequence_length);
  cudaDeviceSynchronize();

  vector<float> q(d_q.number_of_elements_);
  vector<float> k(d_k.number_of_elements_);
  vector<float> v(d_v.number_of_elements_);
  d_q.copy_device_output_to_host(q);
  d_k.copy_device_output_to_host(k);
  d_v.copy_device_output_to_host(v);

  for (int t {0}; t < sequence_length; ++t)
  {
    for (int d {0}; d < kHD; ++d)
    {
      const float row_base {static_cast<float>(t) * 1000.0f};
      EXPECT_FLOAT_EQ(q[t * kHD + d], row_base + d) << "t=" << t << " d=" << d;
      EXPECT_FLOAT_EQ(k[t * kHD + d], row_base + d_model + d)
        << "t=" << t << " d=" << d;
      EXPECT_FLOAT_EQ(v[t * kHD + d], row_base + 2 * d_model + d)
        << "t=" << t << " d=" << d;
    }
  }
}

//------------------------------------------------------------------------------
// General case: B=2, NH=3. Compare against a CPU reference implementing the
// exact index arithmetic documented in split_qkv_heads.h.
//------------------------------------------------------------------------------
TEST(SplitQkvHeadsTests, MatchesCpuReferenceForMultipleBatchesAndHeads)
{
  constexpr int kHD {8};
  constexpr int batch_size {2};
  constexpr int num_heads {3};
  constexpr int sequence_length {5};
  constexpr int d_model {num_heads * kHD};

  const vector<float> qkv {
    make_indexed_qkv(batch_size * sequence_length, 3 * d_model)};

  Array<float> d_qkv(qkv.size());
  Array<float> d_q(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_k(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_v(batch_size * num_heads * sequence_length * kHD);
  d_qkv.copy_host_input_to_device(qkv);

  split_qkv_heads<float, kHD><<<64, 128>>>(
    d_q.elements_, d_k.elements_, d_v.elements_, d_qkv.elements_,
    batch_size, num_heads, sequence_length);
  cudaDeviceSynchronize();

  vector<float> q(d_q.number_of_elements_);
  vector<float> k(d_k.number_of_elements_);
  vector<float> v(d_v.number_of_elements_);
  d_q.copy_device_output_to_host(q);
  d_k.copy_device_output_to_host(k);
  d_v.copy_device_output_to_host(v);

  for (int b {0}; b < batch_size; ++b)
  {
    for (int h {0}; h < num_heads; ++h)
    {
      for (int t {0}; t < sequence_length; ++t)
      {
        for (int d {0}; d < kHD; ++d)
        {
          const int qkv_row {b * sequence_length + t};
          const int out_index {
            ((b * num_heads + h) * sequence_length + t) * kHD + d};

          ASSERT_FLOAT_EQ(
            q[out_index], qkv[qkv_row * 3 * d_model + h * kHD + d])
            << "b=" << b << " h=" << h << " t=" << t << " d=" << d;
          ASSERT_FLOAT_EQ(
            k[out_index],
            qkv[qkv_row * 3 * d_model + d_model + h * kHD + d])
            << "b=" << b << " h=" << h << " t=" << t << " d=" << d;
          ASSERT_FLOAT_EQ(
            v[out_index],
            qkv[qkv_row * 3 * d_model + 2 * d_model + h * kHD + d])
            << "b=" << b << " h=" << h << " t=" << t << " d=" << d;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// merge_heads is the exact inverse of split_qkv_heads's per-tensor gather:
// running split then merge on the Q slice of a fused qkv buffer must
// reconstruct the original Q columns of every row exactly.
//------------------------------------------------------------------------------
TEST(SplitQkvHeadsTests, MergeHeadsInvertsSplit)
{
  constexpr int kHD {4};
  constexpr int batch_size {2};
  constexpr int num_heads {4};
  constexpr int sequence_length {6};
  constexpr int d_model {num_heads * kHD};

  const vector<float> qkv {
    make_indexed_qkv(batch_size * sequence_length, 3 * d_model)};

  Array<float> d_qkv(qkv.size());
  Array<float> d_q(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_k(batch_size * num_heads * sequence_length * kHD);
  Array<float> d_v(batch_size * num_heads * sequence_length * kHD);
  d_qkv.copy_host_input_to_device(qkv);

  split_qkv_heads<float, kHD><<<64, 128>>>(
    d_q.elements_, d_k.elements_, d_v.elements_, d_qkv.elements_,
    batch_size, num_heads, sequence_length);
  cudaDeviceSynchronize();

  Array<float> d_merged(batch_size * sequence_length * d_model);
  merge_heads<float, kHD><<<64, 128>>>(
    d_merged.elements_, d_q.elements_, batch_size, num_heads,
    sequence_length);
  cudaDeviceSynchronize();

  vector<float> merged(d_merged.number_of_elements_);
  d_merged.copy_device_output_to_host(merged);

  // merged[row, :] must equal qkv[row, 0:d_model] -- the Q segment only.
  for (int row {0}; row < batch_size * sequence_length; ++row)
  {
    for (int col {0}; col < d_model; ++col)
    {
      ASSERT_FLOAT_EQ(
        merged[row * d_model + col], qkv[row * 3 * d_model + col])
        << "row=" << row << " col=" << col;
    }
  }
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
