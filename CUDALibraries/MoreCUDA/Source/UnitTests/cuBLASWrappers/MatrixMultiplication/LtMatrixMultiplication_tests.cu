#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtMatrixMultiplication.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "DataStructures/Array.h"
#include "gtest/gtest.h"
#include "StreamManagement/Stream.h"

#include <cstdint>
#include <vector>

using cuBLASWrappers::LibraryContextHandle;
using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::LtHeuristic;
using cuBLASWrappers::MatrixMultiplication::LtLayouts;
using cuBLASWrappers::MatrixMultiplication::LtMatrixMultiplication;
using cuBLASWrappers::MatrixMultiplication::LtPreference;
using cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
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
TEST(cuBLASLtMatmulTests, DefaultConstructs)
{
  LtMatrixMultiplication<float> matmul {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(cuBLASLtMatmulTests, MultiplicationWorksForSmallExample)
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
TEST(cuBLASLtMatmulTests, MultiplicationWorks)
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
TEST(
  cuBLASLtMatmulTests,
  RowMajorAndColumnMajorMultiplicationDoesNotWork)
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
// TEST(cuBLASLtMatmulTests, RowMajorAndColumnMajorMultiplicationWorks)
// {
//   const uint32_t M {4};
//   const uint32_t K {3};
//   const uint32_t N {2};

//   // Matrix A (4x3) in -major order
//   vector<float> A
//   {
//     1, 2, 3,    // row 0
//     4, 5, 6,    // row 1
//     7, 8, 9,    // row 2
//     2, 4, 6     // row 3
//   };

//   // Matrix B (3x2) in row-major order
//   vector<float> B
//   {
//     1, 3, 5,    // column 0
//     2, 4, 6     // column 1
//   };

//   // Matrix C (4x2) in column-major order (same as output D)
//   vector<float> C
//   {
//     1, 2, 3, 4,    // column 0
//     5, 6, 7, 8     // column 1
//   };

//   // Expected result D (4x2) in column-major order: D = A*B + C
//   // A*B calculation:
//   // Row 0: [1,2,3] * [1,2; 3,4; 5,6] = [22,28]
//   // Row 1: [4,5,6] * [1,2; 3,4; 5,6] = [49,64]
//   // Row 2: [7,8,9] * [1,2; 3,4; 5,6] = [76,100]
//   // Row 3: [2,4,6] * [1,2; 3,4; 5,6] = [44,56]
//   vector<float> expected_D
//   {
//     23, 51, 79, 48,    // column 0: A*B(:,0) + C(:,0)
//     33, 70, 107, 64    // column 1: A*B(:,1) + C(:,1)
//   };

//   Array<float> A_array {M * K};
//   Array<float> B_array {K * N};
//   Array<float> C_array {M * N};
//   Array<float> D_array {M * N};

//   EXPECT_TRUE(A_array.copy_host_input_to_device(A));
//   EXPECT_TRUE(B_array.copy_host_input_to_device(B));
//   EXPECT_TRUE(C_array.copy_host_input_to_device(C));

//   Stream stream {};
//   LibraryContextHandle handle {};
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

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests
