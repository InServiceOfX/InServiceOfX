#ifndef OPERATIONS_CONVOLUTION_1D_H
#define OPERATIONS_CONVOLUTION_1D_H

#include <cstddef> // std::size_t
#include <cuda_runtime.h>

namespace Operations
{

//------------------------------------------------------------------------------
/// @details Assume N_b, the size of the convolution, is much less than either
/// N_a or THREADS_PER_BLOCK.
/// 1-dim. convolution can be expressed as
/// out[i] \sum_{j=0}^{N_b - 1} a[i + j] b[j] if i + j < N_a.
//------------------------------------------------------------------------------
template <std::size_t THREADS_PER_BLOCK, std::size_t N_b, typename T>
__global__ void convolve_1d(
  T* output,
  T* a,
  T* b,
  const std::size_t N_a)
{
  // Intentionally show the arithmetic because we are allocating for enough
  // shared memory for the elements of a, one element of a per thread, and
  // N_b - 1 more elements of a for convolution. Then allocate space for b.
  __shared__ T shared_array[THREADS_PER_BLOCK + (N_b - 1) + N_b];

  T* shared_a = shared_array;
  // If THREADS_PER_BLOCK = 4 and N_b = 3, then THREADS_PER_BLOCK + N_b - 1 = 6.
  // 0 1 2 3 | 4 5 | where a[4], a[5] are the boundary values for convolution,
  // and so store the values of b starting from 6.
  T* shared_b = shared_array + THREADS_PER_BLOCK + N_b - 1;
 
  const std::size_t i {blockIdx.x * blockDim.x + threadIdx.x};

  if (i < N_a)
  {
    shared_a[threadIdx.x] = a[i];

    if (threadIdx.x < N_b)
    {
      shared_b[threadIdx.x] = b[threadIdx.x];
    }
    // Here, threadIdx.x >= N_b but threadIdx.x < THREADS_PER_BLOCK, i.e.
    // threadIdx.x = N_b, N_b + 1, ..., THREADS_PER_BLOCK - 1.
    else
    {
      // i_a = threadIdx.x - N_b = 0, 1, ..., THREADS_PER_BLOCK - N_b - 1.
      // i_a has been rescaled to start from 0.
      const std::size_t i_a {threadIdx.x - N_b};
      // i = i_x + M_x * j_x where i_x \equiv threadIdx.x. Observe that
      // 0 + M_x * j_x <= i < (M_x - 1) + M_x * j_x for a single, given
      // thread block.
      // k_a = i - N_b = (i_x - N_b) + M_x * j_x and so given the condition
      // i_x = N_b, N_b, ... M_x - 1, then
      // k_a = M_x * j_x, 1 + M_x * j_x, ... M_x * (j_x + 1) - 1 - N_b.
      const std::size_t k_a {i - N_b};

      // k_a + M_x = M_x + M_x * j_x, (M_x + 1) + M_x * j_x, ..., i.e.
      // k_a + M_x is a rescaled i to start from the first element outside of
      // the current thread block, since it starts at M_x + M_x * j_x
      // (compare this against maximum of i).
      if ((k_a + THREADS_PER_BLOCK < N_a) && (i_a < N_b - 1))
      {
        shared_a[i_a + THREADS_PER_BLOCK] = a[k_a + THREADS_PER_BLOCK];
      }
    }
  }
  __syncthreads();

  T output_value {0};

  for (size_t j {0}; j < N_b; ++j)
  {
    if (i + j < N_a)
    {
      output_value += a[i + j] * b[j];
    }
  }

  output[i] = output_value;
}

} // namespace Operations

#endif // OPERATIONS_CONVOLUTION_1D_H