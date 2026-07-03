#include "DataStructures/Array.h"
#include "Transformer/MultiHeadAttention/tiled_gemm.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using DataStructures::Array;
using std::vector;
using Transformer::MultiHeadAttention::tiled_gemm_launch;

namespace GoogleUnitTests
{
namespace Transformer
{
namespace MultiHeadAttention
{

//------------------------------------------------------------------------------
// Deterministic pseudo-random values in roughly [-1, 1].
//------------------------------------------------------------------------------
vector<float> make_gemm_inputs(const int count, const int seed)
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
// Double-precision CPU reference for row-major Out = X · W.
//------------------------------------------------------------------------------
vector<float> gemm_cpu(
  const vector<float>& input,
  const vector<float>& weight_matrix,
  const int m,
  const int k,
  const int n)
{
  vector<float> output(static_cast<size_t>(m) * n);
  for (int row {0}; row < m; ++row)
  {
    for (int column {0}; column < n; ++column)
    {
      double accumulated {0.0};
      for (int j {0}; j < k; ++j)
      {
        accumulated += static_cast<double>(input[row * k + j]) *
          static_cast<double>(weight_matrix[j * n + column]);
      }
      output[static_cast<size_t>(row) * n + column] =
        static_cast<float>(accumulated);
    }
  }
  return output;
}

void run_and_compare(const int m, const int k, const int n)
{
  const vector<float> input {make_gemm_inputs(m * k, 3)};
  const vector<float> weight_matrix {make_gemm_inputs(k * n, 5)};

  Array<float> d_input(m * k);
  Array<float> d_weights(k * n);
  Array<float> d_output(m * n);
  d_input.copy_host_input_to_device(input);
  d_weights.copy_host_input_to_device(weight_matrix);

  tiled_gemm_launch<float>(
    d_output.elements_, d_input.elements_, d_weights.elements_, m, k, n);
  cudaDeviceSynchronize();

  vector<float> output(static_cast<size_t>(m) * n);
  d_output.copy_device_output_to_host(output);

  const vector<float> expected {gemm_cpu(input, weight_matrix, m, k, n)};
  for (size_t i {0}; i < output.size(); ++i)
  {
    // k up to a few hundred accumulated float terms vs. double reference.
    ASSERT_NEAR(output[i], expected[i], 1e-3f * (1.0f + std::abs(expected[i])))
      << "index " << i;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TiledGemmTests, TileMultipleDimensions)
{
  // All dimensions exact multiples of the 32-wide tile.
  run_and_compare(64, 96, 128);
}

//------------------------------------------------------------------------------
// Ragged sizes exercise every out-of-range guard: partial tiles on the m, k,
// and n edges simultaneously.
//------------------------------------------------------------------------------
TEST(TiledGemmTests, RaggedDimensions)
{
  run_and_compare(33, 65, 47);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TiledGemmTests, SmallerThanOneTile)
{
  run_and_compare(3, 5, 7);
}

//------------------------------------------------------------------------------
// The QKV linear-map shape at small scale: (B·T, d_model)·(d_model, 3·d_model).
//------------------------------------------------------------------------------
TEST(TiledGemmTests, QkvProjectionShape)
{
  run_and_compare(96, 64, 192);
}

} // namespace MultiHeadAttention
} // namespace Transformer
} // namespace GoogleUnitTests
