#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"
#include "cuBLASWrappers/MatrixMultiplication/LtSetDescriptorAttributes.h"
#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

template<typename T>
class Setup
{
  public:

    using LtDescriptor = cuBLASWrappers::MatrixMultiplication::LtDescriptor;
    using LtHeuristic = cuBLASWrappers::MatrixMultiplication::LtHeuristic;
    using LtLayouts = cuBLASWrappers::MatrixMultiplication::LtLayouts;
    using LtPreference = cuBLASWrappers::MatrixMultiplication::LtPreference;
    using LtSetDescriptorAttributes =
      cuBLASWrappers::MatrixMultiplication::LtSetDescriptorAttributes;
    using Workspace = cuBLASWrappers::MatrixMultiplication::Workspace;

    Setup():
      workspace_{},
      descriptor_{get_compute_parameters<T>()},
      set_descriptor_attributes_{},
      layouts_{},
      preference_{},
      heuristic_{}
    {}

    bool setup(cuBLASWrappers::LibraryContextHandle& handle)
    {
      layouts_.create_ABD_layouts<T>();
      layouts_.create_C_layout<T>();

      preference_.set_max_workspace_memory(workspace_.workspace_size_in_bytes_);

      set_descriptor_attributes_.set_epilogue(false, false, false);
      set_descriptor_attributes_.set_epilogue_function(descriptor_.descriptor_);
      set_descriptor_attributes_.set_scale_type(descriptor_.descriptor_);

      heuristic_.get_heuristic(
        handle,
        descriptor_,
        layouts_,
        preference_);

      return true;
    }

    Workspace workspace_;
    LtDescriptor descriptor_;
    LtSetDescriptorAttributes set_descriptor_attributes_;
    LtLayouts layouts_;
    LtPreference preference_;
    LtHeuristic heuristic_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_SETUP_H