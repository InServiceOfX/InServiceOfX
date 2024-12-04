#include "cuBLASWrappers/MatrixMultiplication/cuBLASLtPreference.h"

#include <cstdint>
#include <iostream>
#include <string>

using std::cerr;

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

cuBLASLtPreference::cuBLASLtPreference():
  preference_ {}
{
  handle_create_preference(cublasLtMatmulPreferenceCreate(&preference_));
}

cuBLASLtPreference::~cuBLASLtPreference()
{
  handle_destroy_preference(cublasLtMatmulPreferenceDestroy(preference_));
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulpreferenceattributes-t
/// 3.3.13. cublasLtMatmulPreferenceSetAttributes_t
/// CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES - Max allowed workspace memory.
/// Default is 0 (no workspace memory allowed). uint64_t
//------------------------------------------------------------------------------
bool cuBLASLtPreference::set_max_workspace_memory(
  uint64_t& workspace_size_in_bytes)
{
  const cublasStatus_t status {cublasLtMatmulPreferenceSetAttribute(
    preference_,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size_in_bytes,
    sizeof(workspace_size_in_bytes))};

  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else if (status == CUBLAS_STATUS_INVALID_VALUE)
  {
    std::string error_message {
      "buf is NULL or sizeInBytes doesn't match size of internal storage for "
      "the selected attribute."};

    cerr << error_message << "\n";
    return false;
  }
  else
  {
    cerr << "Failed to set max workspace memory.\n";
    return false;
  }
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulpreferencecreate
//------------------------------------------------------------------------------
bool cuBLASLtPreference::handle_create_preference(const cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_ALLOC_FAILED)
    {
      cerr << "Memory could not be allocated for preference.\n";
    }
    else
    {
      cerr << "Failed to create preference.\n";
    }

    return false;
  }

  return true;
}

bool cuBLASLtPreference::handle_destroy_preference(const cublasStatus_t status)
{
  if (status == CUBLAS_STATUS_SUCCESS)
  {
    return true;
  }
  else
  {
    cerr << "Failed to destroy preference.\n";
    return false;
  }
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers