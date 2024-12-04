#include "cuBLASWrappers/MatrixMultiplication/Workspace.h"

#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

#include <stdexcept>

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace cuBLASWrappers
{
namespace MatrixMultiplication
{

Workspace::Workspace(const size_t workspace_size_in_bytes):
  workspace_size_in_bytes_(workspace_size_in_bytes)
{
  if (workspace_size_in_bytes_ % 256 != 0)
  {
    throw std::invalid_argument(
      "Workspace size must be a multiple of 256 bytes.");
  }

  HandleUnsuccessfulCUDACall handle_malloc {
    "Failed to allocate workspace"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_malloc,
    cudaMalloc(&workspace_, workspace_size_in_bytes_));
}

Workspace::~Workspace()
{
  HandleUnsuccessfulCUDACall handle_free {"Failed to free workspace"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_free,
    cudaFree(workspace_));
}

} // namespace MatrixMultiplication
} // namespace cuBLASWrappers