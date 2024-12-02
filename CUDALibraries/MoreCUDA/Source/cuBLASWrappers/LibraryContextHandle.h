//------------------------------------------------------------------------------
/// See
/// https://docs.nvidia.com/cuda/cublas/#cublaslthandle-t
/// 3.3.3 cublasLtHandle_t
/// cublasLtHandle_t type is a pointer type to an opaque structure holding
/// cuBLASLt library context.
//------------------------------------------------------------------------------

#ifndef CUBLAS_WRAPPERS_LIBRARY_CONTEXT_HANDLE_H
#define CUBLAS_WRAPPERS_LIBRARY_CONTEXT_HANDLE_H

#include <cublasLt.h>

namespace cuBLASWrappers
{

class LibraryContextHandle
{
  public:

    LibraryContextHandle();
    ~LibraryContextHandle();

    // https://docs.nvidia.com/cuda/cublas/#cublasltcreate
    // Pointer to the allocated cuBLASLt handle for the created cuBLASLt
    // context.
    cublasLtHandle_t handle_;

  protected:

    cublasStatus_t create_handle();
    cublasStatus_t destroy_handle();

    bool handle_creation(cublasStatus_t status);
    bool handle_destruction(cublasStatus_t status);
};

} // namespace cuBLASWrappers


#endif // CUBLAS_WRAPPERS_LIBRARY_CONTEXT_HANDLE_H