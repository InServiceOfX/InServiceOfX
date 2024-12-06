#include "StreamManagement/Stream.h"

#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace StreamManagement
{

Stream::Stream()
{
  create_stream();
}

Stream::~Stream()
{
  destroy_stream();
}

bool Stream::create_stream()
{
  HandleUnsuccessfulCUDACall handle_create_stream {"Failed to create stream"};

  //--------------------------------------------------------------------------
  /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da
  /// cudaStreamCreate(cudaStream_t* pStream)
  /// cudaStreamCreate returns
  /// cudaSuccess, cudaErrorInvalidValue
  //--------------------------------------------------------------------------
  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_create_stream,
    cudaStreamCreate(&stream_));

  return handle_create_stream.is_cuda_success();
}

bool Stream::destroy_stream()
{
  HandleUnsuccessfulCUDACall handle_destroy_stream {"Failed to destroy stream"};

  //--------------------------------------------------------------------------
  /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da
  /// cudaStreamDestroy(cudaStream_t stream)
  /// cudaStreamDestroy returns
  /// cudaSuccess, cudaErrorInvalidValue,cudaErrorInvalidResourceHandle
  //--------------------------------------------------------------------------
  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_destroy_stream,
    cudaStreamDestroy(stream_));

  return handle_destroy_stream.is_cuda_success();
}

} // namespace StreamManagement
