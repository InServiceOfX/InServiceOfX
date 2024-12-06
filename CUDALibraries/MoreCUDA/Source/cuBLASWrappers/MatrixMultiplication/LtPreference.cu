#include "cuBLASWrappers/MatrixMultiplication/LtPreference.h"

#include <cstdint>
#include <iostream>

using std::cerr;

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

LtPreference::LtPreference():
  preference_ {}
{
  handle_create_preference(cublasLtMatmulPreferenceCreate(&preference_));
}

LtPreference::~LtPreference()
{
  handle_destroy_preference(cublasLtMatmulPreferenceDestroy(preference_));
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulpreferenceattributes-t
/// 3.3.13. cublasLtMatmulPreferenceSetAttributes_t
/// CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES - Max allowed workspace memory.
/// Default is 0 (no workspace memory allowed). uint64_t
//------------------------------------------------------------------------------
bool LtPreference::set_max_workspace_memory(
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
    static constexpr const char* error_message_1 {
      "buf is NULL or sizeInBytes doesn't match size of internal storage for "};
    static constexpr const char* error_message_2 {
      "the selected attribute."};

    cerr << error_message_1 << error_message_2 << "\n";
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
bool LtPreference::handle_create_preference(const cublasStatus_t status)
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

bool LtPreference::handle_destroy_preference(const cublasStatus_t status)
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