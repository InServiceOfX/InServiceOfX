#ifndef OPERATIONS_DOT_PRODUCT_H
#define OPERATIONS_DOT_PRODUCT_H

#include <cstddef> // std::size_t
#include <cuda_runtime.h>

namespace Operations
{

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/32968071/cuda-dot-product
//------------------------------------------------------------------------------
template <std::size_t THREADS_PER_BLOCK, typename T>
__global__ void dot_product(
  T* input1,
  T* input2,
  T* output,
  const std::size_t size)
{
  __shared__ T shared_array[THREADS_PER_BLOCK];

  const std::size_t index {blockIdx.x * blockDim.x + threadIdx.x};

  if (index < size)
  {
    // 2 global reads.
    shared_array[threadIdx.x] = input1[index] * input2[index];
  }
  
  __syncthreads();

  // Only 1 thread out of the block will be responsible for doing the final
  // summation for the entire thread block.
  if (threadIdx.x == 0)
  {
    T block_sum {static_cast<T>(0)};

    // Sum up multiplied terms within this thread block.
    for (auto i {0}; i < THREADS_PER_BLOCK; ++i)
    {
      if (i + blockIdx.x * blockDim.x < size)
      {
        block_sum += shared_array[i];
      }
    }

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    /// An atomic function performs a read-modify-write atomic operation on 1
    /// 32-bit, 64-bit, or 128-bit word in global or shared memory. In the case
    /// of float2 or float4, read-modify-write operation is performed on each
    /// element of vector residing in global memory. For example, atomicAdd()
    /// reads a word at some address in global or shared memory, adds a number
    /// to it, and writes result back to same address.
    /// More on atomic
    /// https://llvm.org/docs/Atomics.html
    /// https://doc.rust-lang.org/nomicon/atomics.html
    /// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicadd
    /// float atomicAdd(float* address, float val);
    /// The function returns old.
    //--------------------------------------------------------------------------
    // 1 global write.
    atomicAdd(output, block_sum);
  }
}

} // namespace Operations

#endif // OPERATIONS_ARITHMETIC_H
