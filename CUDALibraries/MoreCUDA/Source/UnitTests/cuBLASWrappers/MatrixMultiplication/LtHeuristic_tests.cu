#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"
#include "Utilities/CaptureCerr.h"
#include "gtest/gtest.h"

using cuBLASWrappers::LibraryContextHandle;
using cuBLASWrappers::MatrixMultiplication::LtHeuristic;
using cuBLASWrappers::MatrixMultiplication::LtDescriptor;
using cuBLASWrappers::MatrixMultiplication::LtLayouts;
using cuBLASWrappers::MatrixMultiplication::LtPreference;
using cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
using cuBLASWrappers::MatrixMultiplication::Workspace;

using Utilities::CaptureCerr;

namespace GoogleUnitTests
{
namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtHeuristic, DefaultConstructs)
{
  LtHeuristic heuristic {};

  EXPECT_EQ(heuristic.number_of_algorithms_, 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtHeuristic, GetHeuristicReturnsFalseOnUncreatedInputs)
{
  CaptureCerr capture_cerr {};

  LtHeuristic heuristic {};

  LibraryContextHandle library_context_handle {};
  LtDescriptor descriptor {};
  LtLayouts layouts {};
  LtPreference preference {};

  EXPECT_FALSE(heuristic.get_heuristic(
    library_context_handle,
    descriptor,
    layouts,
    preference));

  EXPECT_EQ(heuristic.number_of_algorithms_, 0);
  EXPECT_EQ(
    capture_cerr.local_oss_.str(),
    "requestedAlgoCount less or equal to 0\n");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LtHeuristic, GetHeuristicWorks)
{
  LtHeuristic heuristic {};

  LibraryContextHandle library_context_handle {};
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

  EXPECT_TRUE(heuristic.get_heuristic(
    library_context_handle,
    descriptor,
    layouts,
    preference));

  EXPECT_EQ(heuristic.number_of_algorithms_, 1);
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers
} // namespace GoogleUnitTests