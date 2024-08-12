#ifndef OPERATIONS_ARITHMETIC_H
#define OPERATIONS_ARITHMETIC_H

#include "DataStructures/Array.h"

#include <cstddef>
#include <cuda_runtime.h>

namespace Operations
{

template <typename T>
__global__ void add_scalar(T* input, T* output, const T scalar)
{
  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  output[index] = input[index] + scalar;
}

template <typename T>
__global__ void add_scalar(
  T* input,
  T* output,
  const std::size_t size,
  const T scalar)
{
  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  if (index < size)
  {
    output[index] = input[index] + scalar;
  }
}

template <typename T>
__global__ void add_scalar_2D(
  T* input,
  T* output,
  const std::size_t row_size,
  const std::size_t column_size,
  const T scalar)
{
  const std::size_t i {blockIdx.x * blockDim.x + threadIdx.x};
  const std::size_t j {blockIdx.y * blockDim.y + threadIdx.y};
  if (i < row_size && j < column_size)
  {
    const std::size_t index {i + j * row_size};
    output[index] = input[index] + scalar;
  }
}

template <typename T>
__global__ void addition(const T* addend1, const T* addend2, T* sum)
{
  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  sum[index] = addend1[index] + addend2[index];
}

template <typename T>
__global__ void addition(
  const T* addend1,
  const T* addend2,
  T* sum,
  const std::size_t size)
{
  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};
  if (index < size)
  {
    sum[index] = addend1[index] + addend2[index];
  }
}

} // namespace Operations

#endif // OPERATIONS_ARITHMETIC_H
