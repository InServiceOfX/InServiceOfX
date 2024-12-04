#ifndef CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_WORKSPACE_H
#define CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_WORKSPACE_H

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

//------------------------------------------------------------------------------
/// From
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublasltmatmul
/// 3.4.17. cublasLtMatmul()
/// cublasStatus_t cublasLtMatmul(..
///   void *workspace,
///   size_t workspaceSizeInBytes,..)
/// "The workspace pointer must be aligned to at least a multiple of 256 bytes"
/// The recommendations on workspaceSizeInBytes are same as mentioned in
/// cublasSetWorkspace() section.
/// https://docs.nvidia.com/cuda/cublas/index.html?highlight=CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES#cublassetworkspace
/// 2.4.8 cublasSetWorkspace()
/// GPU Architecture, Recommended workspace size
/// NVIDIA Hopper Architecture, 32 MiB
/// Other, 4 MiB
//------------------------------------------------------------------------------
class Workspace
{
  public:
  
    Workspace(const size_t workspace_size_in_bytes=4 * 1024 * 1024);

    ~Workspace();

    void* workspace_;

    size_t workspace_size_in_bytes_;
};

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers

#endif // CUBLAS_WRAPPERS_MATRIX_MULTIPLICATION_WORKSPACE_H