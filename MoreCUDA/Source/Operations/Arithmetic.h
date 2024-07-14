#ifndef OPERATIONS_ARITHMETIC_H
#define OPERATIONS_ARITHMETIC_H

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

} // namespace Operations

#endif // OPERATIONS_ARITHMETIC_H
