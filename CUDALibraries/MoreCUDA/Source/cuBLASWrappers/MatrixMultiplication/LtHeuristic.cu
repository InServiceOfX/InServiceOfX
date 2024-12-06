#include "cuBLASWrappers/MatrixMultiplication/LtHeuristic.h"

#include "cuBLASWrappers/LibraryContextHandle.h"
#include "cuBLASWrappers/MatrixMultiplication/LtDescriptor.h"
#include "cuBLASWrappers/MatrixMultiplication/LtLayouts.h"
#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"

#include <iostream>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulalgogetheuristic
/// 3.4.22. cublasLtMatmulAlgoGetHeuristic()
/// cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
///   cublasLtHandle_t lightHandle,
///   cublasLtMatmulDesc_t operationDesc,
///   cublasLtMatrixLayout_t Adesc,
///   ...);
//------------------------------------------------------------------------------
bool LtHeuristic::get_heuristic(
  cuBLASWrappers::LibraryContextHandle& library_context_handle,
  LtDescriptor& descriptor,
  LtLayouts& layouts,
  LtPreference& preference,
  const int requested_algorithm_count)
{
  cublasStatus_t status {cublasLtMatmulAlgoGetHeuristic(
    library_context_handle.handle_,
    descriptor.descriptor_,
    layouts.A_layout_,
    layouts.B_layout_,
    layouts.C_layout_,
    layouts.D_layout_,
    preference.preference_,
    requested_algorithm_count,
    &heuristic_,
    &number_of_algorithms_)};

  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else if (status == CUBLAS_STATUS_NOT_SUPPORTED)
  {
    std::cerr << "no heuristic function available for current configuration.\n";
    return false;
  }
  else if (status == CUBLAS_STATUS_INVALID_VALUE)
  {
    std::cerr << "requestedAlgoCount less or equal to 0\n";
    return false;
  }
  else
  {
    std::cerr << "cublasLtMatmulAlgoGetHeuristic() failed with status "
      << status << std::endl;
    return false;
  }
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers