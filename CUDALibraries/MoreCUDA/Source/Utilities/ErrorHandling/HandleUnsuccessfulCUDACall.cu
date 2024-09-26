#include "HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>
#include <iostream> // std::cerr
#include <string>
#include <string_view>

using std::cerr;

namespace Utilities
{
namespace ErrorHandling
{

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const std::string_view error_message
  ):
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const std::string& error_message
  ):
  // Implicit conversion from std::string to std::string_view
  // https://en.cppreference.com/w/cpp/string/basic_string/operator_basic_string_view
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}

HandleUnsuccessfulCUDACall::HandleUnsuccessfulCUDACall(
  const char* error_message
  ):
  error_message_{error_message},
  cuda_error_{cudaSuccess}
{}

void HandleUnsuccessfulCUDACall::operator()(const cudaError_t cuda_error)
{
  cuda_error_ = cuda_error;

  if (!is_cuda_success())
  {
    cerr << error_message_ << " (error code " <<
      cudaGetErrorString(cuda_error_) << ")!\n";
  }
}

void HandleUnsuccessfulCUDACall::operator()(
  const cudaError_t cuda_error,
  const char* file,
  const int line)
{
  cuda_error_ = cuda_error;

  if (!is_cuda_success())
  {
    cerr << file << ":" << line << ": " << error_message_ << " (error code " <<
      cudaGetErrorString(cuda_error_) << ")!\n";
  }
}

} // namespace ErrorHandling
} // namespace Utilities