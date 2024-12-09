#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "cuBLASWrappers/MatrixMultiplication/Setup.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"
#include "StreamManagement/Stream.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using cuBLASWrappers::MatrixMultiplication::general_matrix_multiply;
using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::LtHeuristic;
using cuBLASWrappers::MatrixMultiplication::LtLayouts;
using cuBLASWrappers::MatrixMultiplication::LtMatrixMultiplication;
using cuBLASWrappers::MatrixMultiplication::LtPreference;
using cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
using cuBLASWrappers::MatrixMultiplication::matrix_multiply;
using cuBLASWrappers::MatrixMultiplication::Setup;
using cuBLASWrappers::MatrixMultiplication::Workspace;
using DataStructures::Array;
using StreamManagement::Stream;
using std::vector;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

float gelu(const float x)
{
  constexpr float sqrt_2_over_pi {0.7978845608028654f};
  constexpr float coef {0.044715f};
  const float x_cubed {x * x * x};
  const float tanh_input {sqrt_2_over_pi * (x + coef * x_cubed)};
  return 0.5f * x * (1.0f + std::tanh(tanh_input));
}

vector<float> apply_gelu(const vector<float>& input)
{
  vector<float> result {};
  result.reserve(input.size());

  std::transform(
    input.begin(),
    input.end(),
    std::back_inserter(result),
    [](const float x) { return gelu(x); }
  );

  return result;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MinimalSetup)
{
  Stream stream {};
  LibraryContextHandle handle {};
  Workspace workspace {};
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
  LtLayouts layouts {8, 4, 2};
  EXPECT_TRUE(layouts.create_ABD_layouts<float>());
  EXPECT_TRUE(layouts.create_C_layout<float>());

  LtPreference preference {};
  EXPECT_TRUE(preference.set_max_workspace_memory(
    workspace.workspace_size_in_bytes_));

  set_descriptor_attributes.set_epilogue(false, false, false);
  EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_scale_type(
    descriptor.descriptor_));

  LtHeuristic heuristic {};
  EXPECT_TRUE(heuristic.get_heuristic(
    handle,
    descriptor,
    layouts,
    preference));

  EXPECT_EQ(heuristic.number_of_algorithms_, 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtMatrixMultiplicationTests, DefaultConstructs)
{
  LtMatrixMultiplication<float> matmul {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtMatrixMultiplicationTests, MultiplicationWorksForSmallExample)
{
  const uint32_t M {4};
  const uint32_t K {2};
  const uint32_t N {3};

  // For some reason, by default, matrices are column-major in cuBLASLt.
  // Matrix A (4x2)
  vector<float> A
  {
    1, 0, 2, 3,    // column 0
    1, 1, 4, 5     // column 1
  };

  // Matrix B (2x3)
  vector<float> B
  {
    1, 2,     // column 0
    3, 4,     // column 1
    5, 6      // column 2
  };

  // Expected result D (4x3)
  vector<float> expected_D
  {
    3, 2, 10, 13,  // column 0
    7, 4, 22, 29,  // column 1
    11, 6, 34, 45   // column 2
  };

  Array<float> A_array {M * K};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  Stream stream {};
  LibraryContextHandle handle {};
  Workspace workspace {};
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
  LtLayouts layouts {M, N, K};
  EXPECT_TRUE(layouts.create_ABD_layouts<float>());
  EXPECT_TRUE(layouts.create_C_layout<float>());

  LtPreference preference {};
  EXPECT_TRUE(preference.set_max_workspace_memory(
    workspace.workspace_size_in_bytes_));

  set_descriptor_attributes.set_epilogue(false, false, false);
  EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_scale_type(
    descriptor.descriptor_));

  LtHeuristic heuristic {};
  EXPECT_TRUE(heuristic.get_heuristic(
    handle,
    descriptor,
    layouts,
    preference));

  LtMatrixMultiplication<float> matrix_multiplication {};
  EXPECT_TRUE(matrix_multiplication(
    handle,
    descriptor,
    layouts,
    heuristic,
    stream,
    workspace,
    A_array.elements_,
    B_array.elements_,
    nullptr,
    D_array.elements_));

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtMatrixMultiplicationTests, MultiplicationWorks)
{
  const uint32_t M {8};
  const uint32_t K {2};
  const uint32_t N {4};

  // Matrix A (8x2) in column-major order
  vector<float> A
  {
    // First column (8 elements)
    1, 3, 1, 0, 2, 1, 3, 2,    // column 0
    // Second column (8 elements)
    2, 4, 0, 1, 1, 1, 2, 3     // column 1
  };

  // Matrix B (2x4) in column-major order
  vector<float> B
  {
    1, 5,     // column 0
    2, 6,     // column 1
    3, 7,     // column 2
    4, 8      // column 3
  };

  // Expected result D (8x4) in column-major order
  vector<float> expected_D
  {
    // Each column has 8 elements
    11, 23, 1, 5, 7, 6, 13, 17,     // column 0
    14, 30, 2, 6, 10, 8, 18, 22,    // column 1
    17, 37, 3, 7, 13, 10, 23, 27,   // column 2
    20, 44, 4, 8, 16, 12, 28, 32    // column 3
  };

  Array<float> A_array {M * K};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  Stream stream {};
  LibraryContextHandle handle {};
  Workspace workspace {};
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
  LtLayouts layouts {M, N, K};
  EXPECT_TRUE(layouts.create_ABD_layouts<float>());
  EXPECT_TRUE(layouts.create_C_layout<float>());

  LtPreference preference {};
  EXPECT_TRUE(preference.set_max_workspace_memory(
    workspace.workspace_size_in_bytes_));

  set_descriptor_attributes.set_epilogue(false, false, false);
  EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_scale_type(
    descriptor.descriptor_));

  LtHeuristic heuristic {};
  EXPECT_TRUE(heuristic.get_heuristic(
    handle,
    descriptor,
    layouts,
    preference));

  LtMatrixMultiplication<float> matrix_multiplication {};
  EXPECT_TRUE(matrix_multiplication(
    handle,
    descriptor,
    layouts,
    heuristic,
    stream,
    workspace,
    A_array.elements_,
    B_array.elements_,
    nullptr,
    D_array.elements_));

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtMatrixMultiplicationTests, RowMajorAndColumnMajorMultiplicationDoesNotWork)
{
  const uint32_t M {4};
  const uint32_t K {3};
  const uint32_t N {2};

  // Matrix A (4x3) in -major order
  vector<float> A
  {
    1, 2, 3,    // row 0
    4, 5, 6,    // row 1
    7, 8, 9,    // row 2
    2, 4, 6     // row 3
  };

  // Matrix B (3x2) in row-major order
  vector<float> B
  {
    1, 3, 5,    // column 0
    2, 4, 6     // column 1
  };

  // Expected result D (4x2) in column-major order: D = A*B
  vector<float> expected_D
  {
    22, 49, 76, 44,    // column 0: A*B(:,0)
    28, 64, 100, 56    // column 1: A*B(:,1)
  };

  Array<float> A_array {M * K};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  Stream stream {};
  LibraryContextHandle handle {};
  Workspace workspace {};
  LtDescriptor descriptor {};
  LtSetDescriptorAttributes set_descriptor_attributes {};
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
    descriptor.descriptor_));
  LtLayouts layouts {M, N, K};
  EXPECT_TRUE(layouts.create_ABD_layouts<float>());
  EXPECT_TRUE(layouts.create_C_layout<float>());
  EXPECT_TRUE(layouts.set_memory_order('A', CUBLASLT_ORDER_ROW));

  LtPreference preference {};
  EXPECT_TRUE(preference.set_max_workspace_memory(
    workspace.workspace_size_in_bytes_));

  set_descriptor_attributes.set_epilogue(false, false, false);
  EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
    descriptor.descriptor_));
  EXPECT_TRUE(set_descriptor_attributes.set_scale_type(
    descriptor.descriptor_));

  LtHeuristic heuristic {};
  EXPECT_TRUE(heuristic.get_heuristic(
    handle,
    descriptor,
    layouts,
    preference));

  LtMatrixMultiplication<float> matrix_multiplication {1.0};
  EXPECT_TRUE(matrix_multiplication(
    handle,
    descriptor,
    layouts,
    heuristic,
    stream,
    workspace,
    A_array.elements_,
    B_array.elements_,
    nullptr,
    D_array.elements_));

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_NE(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, RowMajorAndRowMajorMultiplicationWorks)
{
  const uint32_t M {4};
  const uint32_t K {3};
  const uint32_t N {2};

  // Matrix A (4x3) in row-major order
  vector<float> A
  {
    1, 2, 3,    // row 0
    4, 5, 6,    // row 1
    7, 8, 9,    // row 2
    2, 4, 6     // row 3
  };

  // Matrix B (3x2) in row-major order
  vector<float> B
  {
    1, 2,
    3, 4,
    5, 6
  };

  vector<float> expected_D
  {
    23, 51, 79, 48,    // column 0: A*B(:,0)
    33, 70, 107, 64    // column 1: A*B(:,1)
  };

  Array<float> A_array {M * K};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  Stream stream {};
  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {};
}

//   Workspace workspace {};
//   LtDescriptor descriptor {};
//   LtSetDescriptorAttributes set_descriptor_attributes {};
//   EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_A(
//     descriptor.descriptor_));
//   EXPECT_TRUE(set_descriptor_attributes.set_transpose_on_B(
//     descriptor.descriptor_));
//   LtLayouts layouts {M, N, K};
//   EXPECT_TRUE(layouts.create_ABD_layouts<float>());
//   EXPECT_TRUE(layouts.create_C_layout<float>());
//   EXPECT_TRUE(layouts.set_memory_order('A', CUBLASLT_ORDER_ROW));

//   LtPreference preference {};
//   EXPECT_TRUE(preference.set_max_workspace_memory(
//     workspace.workspace_size_in_bytes_));

//   set_descriptor_attributes.set_epilogue(false, false, false);
//   EXPECT_TRUE(set_descriptor_attributes.set_epilogue_function(
//     descriptor.descriptor_));
//   EXPECT_TRUE(set_descriptor_attributes.set_scale_type(
//     descriptor.descriptor_));

//   LtHeuristic heuristic {};
//   EXPECT_TRUE(heuristic.get_heuristic(
//     handle,
//     descriptor,
//     layouts,
//     preference));

//   LtMatrixMultiplication<float> matrix_multiplication {1.0, 1.0};
//   EXPECT_TRUE(matrix_multiplication(
//     handle,
//     descriptor,
//     layouts,
//     heuristic,
//     stream,
//     workspace,
//     A_array.elements_,
//     B_array.elements_,
//     nullptr,
//     D_array.elements_));

//   vector<float> resulting_D (M * N);
//   EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

//   EXPECT_EQ(resulting_D, expected_D);
// }

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MultiplicationWithTransposeOnAWorks)
{
  const uint32_t M {3};  // rows of result and A^T
  const uint32_t N {4};  // cols of B and result
  const uint32_t K {2};  // cols of A^T, rows of B

  // Matrix A (2x3) in column-major order, will be transposed to (3x2)
  // Using powers of 2 for exact binary representation
  vector<float> A
  {
    0.5f,  2.0f,    // column 0
    1.0f,  4.0f,    // column 1
    0.25f, 0.125f   // column 2
  };
  // When transposed becomes:
  // [0.5  1.0   0.25]
  // [2.0  4.0   0.125]

  // Matrix B (2x4) in column-major order
  vector<float> B
  {
    1.0f, 0.5f,     // column 0
    2.0f, 1.0f,     // column 1
    4.0f, 2.0f,     // column 2
    8.0f, 4.0f      // column 3
  };

  Array<float> A_array {K * M};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  vector<float> expected_D
  {
    1.5f,   3.0f, 0.3125f,    // column 0
    3.f,    6.f,  0.625f,     // column 1
    6.0f,    12.f,  1.25f,      // column 2
    12.0f,   24.f,   2.5f       // column 3
  };

  Stream stream {};
  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {M, N, K};

  EXPECT_TRUE(setup.setup(handle, true));
  const auto transpose_operation_on_A {
    setup.set_descriptor_attributes_.get_transpose_operation_on_A(
      setup.descriptor_.descriptor_)};

  EXPECT_TRUE(transpose_operation_on_A);
  EXPECT_EQ(transpose_operation_on_A->first, static_cast<int32_t>(CUBLAS_OP_T));

  LtMatrixMultiplication<float> matrix_multiplication {};
  EXPECT_TRUE(matrix_multiply(
    handle,
    stream,
    setup,
    matrix_multiplication,
    A_array,
    B_array,
    D_array));

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, BatchedMultiplicationWorks)
{
  const uint32_t M {3};  // rows of A and result
  const uint32_t N {4};  // cols of B and result
  const uint32_t K {2};  // cols of A, rows of B
  const uint32_t batch_count {2};

  // Two A matrices (3x2) in column-major order
  vector<float> A
  {
    // First matrix A1
    0.5f, -1.0f, 2.0f,   // column 0
    0.25f, 0.5f, -0.5f,  // column 1
    
    // Second matrix A2
    -2.0f, 1.0f, 0.5f,   // column 0
    0.5f, -0.25f, 0.125f // column 1
  };

  // Two B matrices (2x4) in column-major order
  vector<float> B
  {
    // First matrix B1
    1.0f, -0.5f,         // column 0
    0.25f, 0.5f,         // column 1
    -1.0f, 2.0f,         // column 2
    0.5f, -1.0f,         // column 3
    
    // Second matrix B2
    -0.5f, 1.0f,         // column 0
    2.0f, -0.25f,        // column 1
    0.125f, 0.5f,        // column 2
    -1.0f, 0.25f         // column 3
  };

  // Expected results D (3x4) in column-major order
  vector<float> expected_D
  {
    // First result D1 = A1 * B1
    0.375f, -1.25f, 2.25f,     // column 0
    0.25f, 0.f, 0.25f,       // column 1
    0.0f, 2.f, -3.f,        // column 2
    0.f, -1.f, 1.5f,     // column 3

    // Second result D2 = A2 * B2
    1.5f, -0.75f, -0.125f,   // column 0
    -4.125f, 2.0625f, 0.96875f,      // column 1
    -0.0f, 0.f, 0.125f,    // column 2
    2.125f, -1.0625f, -0.46875f     // column 3
  };

  Array<float> A_array {M * K * batch_count};
  Array<float> B_array {K * N * batch_count};
  Array<float> D_array {M * N * batch_count};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  Stream stream {};
  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {M, N, K};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float>::BatchCountAndStridedOffsets
    batch_count_and_strided_offsets {2};
  batch_count_and_strided_offsets.set_strided_offsets(M, N, K);

  EXPECT_TRUE(
    setup.setup(handle, false, false, batch_count_and_strided_offsets));

  LtMatrixMultiplication<float> matrix_multiplication {};
  EXPECT_TRUE(matrix_multiply(
    handle,
    stream,
    setup,
    matrix_multiplication,
    A_array,
    B_array,
    D_array));

  vector<float> resulting_D (M * N * batch_count);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MatrixMultiplicationWithBiasAdditionWorks)
{
  const uint32_t M {3};  // rows of A and result
  const uint32_t N {4};  // cols of B and result
  const uint32_t K {2};  // cols of A, rows of B

  // Matrix A (3x2) in column-major order
  vector<float> A
  {
    0.5f, -1.0f, 2.0f,    // column 0
    0.25f, 0.5f, -0.5f    // column 1
  };

  // Matrix B (2x4) in column-major order
  vector<float> B
  {
    1.0f, -0.5f,          // column 0
    2.0f, 1.0f,           // column 1
    -1.0f, 0.5f,          // column 2
    0.5f, -1.0f           // column 3
  };

  // Bias matrix (3x4) in column-major order
  vector<float> C
  {
    0.5f, -0.5f, 1.0f,    // column 0
    0.25f, 0.25f, -0.25f, // column 1
    -0.5f, 0.5f, 0.0f,    // column 2
    1.0f, -1.0f, 0.5f     // column 3
  };

  // Expected D = A*B + bias (3x4) in column-major order
  vector<float> expected_D
  {
    0.875f, -1.75f, 3.25f,    // column 0
    1.5f, -1.25f, 3.25f,  // column 1
    -0.875f, 1.75f, -2.25f,    // column 2
    1.0f, -2.0f, 2.0f    // column 3
  };

  Array<float> A_array {M * K};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));
  EXPECT_TRUE(D_array.copy_host_input_to_device(C));

  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {M, N, K};

  DataStructures::Array<float> bias {
    static_cast<uint32_t>(
      setup.set_descriptor_attributes_.get_bias_size(3, 4))};

  EXPECT_TRUE(setup.setup(handle, false, false, std::nullopt, bias.elements_));

  Stream stream {};
  LtMatrixMultiplication<float> matrix_multiplication {1.0, 1.0};
  EXPECT_TRUE(general_matrix_multiply(
    handle,
    stream,
    setup,
    matrix_multiplication,
    A_array,
    B_array,
    D_array));

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MatrixMultiplicationWithBiasAndBatchesWorks)
{
  const uint32_t M {3};  // rows of A and result
  const uint32_t N {4};  // cols of B and result
  const uint32_t K {2};  // cols of A, rows of B
  const uint32_t batch_count {2};

  // Two A matrices (3x2) in column-major order
  vector<float> A
  {
    // First A1
    0.5f, -1.0f, 2.0f,     // column 0
    0.25f, 0.5f, -0.5f,    // column 1
    
    // Second A2
    -0.5f, 1.0f, 0.25f,    // column 0
    0.5f, -0.25f, 0.125f   // column 1
  };

  // Two B matrices (2x4) in column-major order
  vector<float> B
  {
    // First B1
    1.0f, -0.5f,           // column 0
    2.0f, 1.0f,            // column 1
    -1.0f, 0.5f,           // column 2
    0.5f, -1.0f,           // column 3
    
    // Second B2
    -0.5f, 1.0f,           // column 0
    0.25f, -0.5f,          // column 1
    0.5f, -1.0f,           // column 2
    -0.25f, 0.5f           // column 3
  };

  // Two bias matrices (3x4) in column-major order
  vector<float> C
  {
    // First bias1
    0.5f, -0.5f, 1.0f,     // column 0
    0.25f, 0.25f, -0.25f,  // column 1
    -0.5f, 0.5f, 0.0f,     // column 2
    1.0f, -1.0f, 0.5f,     // column 3
    
    // Second bias2
    -0.25f, 0.5f, -0.125f, // column 0
    0.5f, -0.25f, 0.125f,  // column 1
    0.25f, -0.5f, 0.0f,    // column 2
    -0.5f, 0.25f, -0.125f  // column 3
  };

  // Expected results D (3x4) in column-major order
  vector<float> expected_D
  {
    // First D1 = A1*B1 + bias1
    0.875f, -1.75f, 3.25f,     // column 0
    1.5f, -1.25f, 3.25f,   // column 1
    -0.875f, 1.75f, -2.25f,     // column 2
    1.0f, -2.0f, 2.0f,    // column 3
    
    // Second D2 = A2*B2 + bias2
    0.5f, -0.25f, -0.125f,     // column 0
    0.125f, 0.125f, 0.125f,   // column 1
    -0.5f, 0.25f, 0.0f,   // column 2
    -0.125f, -0.125f, -0.125f   // column 3
  };

  Array<float> A_array {M * K * batch_count};
  Array<float> B_array {K * N * batch_count};
  Array<float> D_array {M * N * batch_count};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));
  EXPECT_TRUE(D_array.copy_host_input_to_device(C));

  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {M, N, K};

  ::cuBLASWrappers::MatrixMultiplication::Setup<float>::BatchCountAndStridedOffsets
    batch_count_and_strided_offsets {batch_count};
  batch_count_and_strided_offsets.set_strided_offsets(M, N, K);

  DataStructures::Array<float> bias {
    static_cast<uint32_t>(
      setup.set_descriptor_attributes_.get_bias_size(3, 4))};

  EXPECT_TRUE(
    setup.setup(
      handle,
      false,
      false,
      batch_count_and_strided_offsets,
      bias.elements_));

  Stream stream {};
  LtMatrixMultiplication<float> matrix_multiplication {1.0, 1.0};

  EXPECT_TRUE(general_matrix_multiply(
    handle,
    stream,
    setup,
    matrix_multiplication,
    A_array,
    B_array,
    D_array));

  vector<float> resulting_D (M * N * batch_count);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  EXPECT_EQ(resulting_D, expected_D);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MatrixMultiplicationWithGELUWorks)
{
  const uint32_t M {8};
  const uint32_t N {3};
  const uint32_t K {2};

  // Matrix A (8x2) in column-major order
  vector<float> A
  {
    // First column (8 elements)
    2.0f, -1.0f, 0.5f, 0.0f, -2.0f, 1.0f, -0.5f, 0.25f,    // column 0
    1.0f, 0.5f, -0.25f, 0.125f, 0.0f, -1.0f, 2.0f, -0.5f   // column 1
  };

  // Matrix B (2x3) in column-major order
  vector<float> B
  {
    0.5f, 1.0f,     // column 0
    -0.5f, 0.25f,   // column 1
    1.0f, -0.5f     // column 2
  };

  Array<float> A_array {K * M};
  Array<float> B_array {K * N};
  Array<float> D_array {M * N};
  Array<float> pre_gelu {M * N};

  EXPECT_TRUE(A_array.copy_host_input_to_device(A));
  EXPECT_TRUE(B_array.copy_host_input_to_device(B));

  LibraryContextHandle handle {};
  ::cuBLASWrappers::MatrixMultiplication::Setup<float> setup {M, N, K};
  EXPECT_TRUE(setup.setup_with_gelu(handle, pre_gelu.elements_));

  Stream stream {};
  LtMatrixMultiplication<float> matrix_multiplication {};
  EXPECT_TRUE(matrix_multiply(
    handle,
    stream,
    setup,
    matrix_multiplication,
    A_array,
    B_array,
    D_array));

  // Expected result before GELU (8x3) in column-major order
  vector<float> expected_pre_gelu
  {
    2.0f, 0.0f, 0.0f, 0.125f, -1.0f, -0.5f, 1.75f, -0.375f,     // column 0
    -0.75f, 0.625f, -0.3125f, 0.03125f, 1.0f, -0.75f, 0.75f, -0.25f,
    1.5f, -1.25f, 0.625f, -0.0625f, -2.0f, 1.5f, -1.5f, 0.5f    // column 2
  };

  vector<float> expected_D {apply_gelu(expected_pre_gelu)};

  vector<float> resulting_D (M * N);
  EXPECT_TRUE(D_array.copy_device_output_to_host(resulting_D));

  for (std::size_t i {0}; i < M * N; ++i)
  {
    EXPECT_NEAR(resulting_D[i], expected_D[i], 1e-4f);
  }

  vector<float> resulting_pre_gelu (M * N);
  EXPECT_TRUE(pre_gelu.copy_device_output_to_host(resulting_pre_gelu));

  EXPECT_EQ(resulting_pre_gelu, expected_pre_gelu);
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
