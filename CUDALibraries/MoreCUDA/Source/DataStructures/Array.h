#ifndef DATA_STRUCTURES_ARRAY_H
#define DATA_STRUCTURES_ARRAY_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace DataStructures
{

template <typename T>
struct Array
{
  using HandleUnsuccessfulCUDACall =
    Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

  T* elements_;
  // Because for CUDA, uint1 is of size 4 bytes=32 bits, use uint32_t. See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3
  const uint32_t number_of_elements_;
  bool is_cuda_freed_;

  Array(const uint32_t number_of_elements = 50000):
    elements_{nullptr},
    number_of_elements_{number_of_elements},
    is_cuda_freed_{false}
  {
    const uint32_t size_in_bytes {
      number_of_elements * static_cast<uint32_t>(sizeof(T))};
    HandleUnsuccessfulCUDACall handle_malloc {
      "Failed to allocate device array"};

    //--------------------------------------------------------------------------
    /// While the function signature of cudaMalloc(..) is
    /// __host__ ​ __device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size ) 
    /// size_t could be an unsigned int in CUDA and wouldn't be the 64-bit x86
    /// architecture, and so use uint32_t. See
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356
    //--------------------------------------------------------------------------
    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_malloc,
      cudaMalloc(reinterpret_cast<void**>(&elements_), size_in_bytes));

    if (!handle_malloc.is_cuda_success())
    {
      throw std::runtime_error(std::string{handle_malloc.get_error_message()});
    }
  }

  ~Array()
  {
    free_resources();
  }

  bool copy_host_input_to_device(const std::vector<T>& h_a)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    //--------------------------------------------------------------------------
    /// While the function signature of cudaMemcpy(..) is
    /// __host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) 
    /// size_t could be an unsigned int in CUDA and wouldn't be the 64-bit x86
    /// architecture, and so use uint32_t. See
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8
    //--------------------------------------------------------------------------
    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_values,
      cudaMemcpy(
        elements_,
        h_a.data(),
        static_cast<uint32_t>(h_a.size() * sizeof(T)),
        cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();
  }

  bool copy_host_input_to_device(const T* h_a, const uint32_t size_in_bytes)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_values,
      cudaMemcpy(
        elements_,
        h_a,
        size_in_bytes,
        cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();
  }

  bool copy_device_output_to_host(std::vector<T>& h_a)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from device to host"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_values,
      cudaMemcpy(
        h_a.data(),
        elements_,
        number_of_elements_ * static_cast<uint32_t>(sizeof(T)),
        cudaMemcpyDeviceToHost));

    return handle_values.is_cuda_success();
  }

  bool copy_device_output_to_host(T* host_ptr)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from device to host"};

    HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
      handle_values,
      cudaMemcpy(
        host_ptr,
        elements_,
        number_of_elements_ * static_cast<uint32_t>(sizeof(T)),
        cudaMemcpyDeviceToHost));

    return handle_values.is_cuda_success();
  }

  bool free_resources()
  {
    if ((elements_ != nullptr) && (is_cuda_freed_ == false))
    {
      HandleUnsuccessfulCUDACall handle_free {"Failed to free device array"};

      HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
        handle_free,
        cudaFree(elements_));

      if (!handle_free.is_cuda_success())
      {
        std::cerr << handle_free.get_error_message() << "\n";
      }
      else
      {
        is_cuda_freed_ = true;
      }

      return handle_free.is_cuda_success();
    }

    return false;
  }
};

} // namespace DataStructures

#endif // DATA_STRUCTURES_ARRAY_H