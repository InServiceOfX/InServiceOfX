#include "cuBLASWrappers/LibraryContextHandle.h"

#include <cublasLt.h>
#include <iostream> // std::cerr
#include <stdexcept>
#include <string>

using std::cerr;
using std::string;

namespace cuBLASWrappers
{

LibraryContextHandle::LibraryContextHandle():
  handle_{}
{
  handle_creation(create_handle());
}

LibraryContextHandle::~LibraryContextHandle()
{
  handle_destruction(destroy_handle());
}

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltcreate
/// cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle)
//------------------------------------------------------------------------------
cublasStatus_t LibraryContextHandle::create_handle()
{
  return cublasLtCreate(&handle_);
}
//------------------------------------------------------------------------------
/// https://docs.nvidia.com/cuda/cublas/#cublasltdestroy
/// cublasStatus_t cublasLtDestroy(cublasLtHandle_t lighthandle)
//------------------------------------------------------------------------------
cublasStatus_t LibraryContextHandle::destroy_handle()
{
  return cublasLtDestroy(handle_);
}

// https://docs.nvidia.com/cuda/cublas/#cublasltcreate
bool LibraryContextHandle::handle_creation(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED)
    {
      std::string error_message {
        "cuBLASLt library was not initialized. This usually happens: error in "
        "CUDA Runtime API, or error in hardware setup."};
      
      throw std::runtime_error(error_message);
    }
    else if (status == CUBLAS_STATUS_ALLOC_FAILED)
    {
      std::string error_message {
        "Resource allocation failed inside cuBLASLt library. This is usually "
        "caused by a cudaMalloc() failure.\n\nTo correct: prior to function "
        "call, deallocate previously allocated memory as much as possible."};

      throw std::runtime_error(error_message);
    }
    else if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      throw std::runtime_error("lighthandle == NULL");
    }
    else
    {
      throw std::runtime_error("Unknown cuBLASLt library error.");
    }
  }

  return status == CUBLAS_STATUS_SUCCESS;
}

bool LibraryContextHandle::handle_destruction(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED)
    {
      static constexpr const char* error_message {
        "cuBLASLt library was not initialized."};

      cerr << error_message << '\n';
    }
    else if (status == CUBLAS_STATUS_INVALID_VALUE)
    {
      cerr << "lighthandle == NULL" << '\n';
    }
    else
    {
      cerr << "Unknown cuBLASLt library error.\n";
    }
  }

  return status == CUBLAS_STATUS_SUCCESS;
}

} // namespace cuBLASWrappers
