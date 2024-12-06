#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_PREFERENCE_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_PREFERENCE_H

#include <cstdint>
#include <cublasLt.h>

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmulpreference-t
/// 3.3.12.cublasLtMatmulPreference_t
/// cublasMatmulPreference_t is a pointer to an opaque structure holding the
/// description of the preferences for cublasLtMatmulAlgoGetHeuristic()
/// configuration.
//------------------------------------------------------------------------------
class LtPreference
{
  public:

    LtPreference();

    ~LtPreference();

    bool set_max_workspace_memory(uint64_t& workspace_size_in_bytes);

    cublasLtMatmulPreference_t preference_;

  private:

    bool handle_create_preference(const cublasStatus_t status);
    bool handle_destroy_preference(const cublasStatus_t status);
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_LT_PREFERENCE_H