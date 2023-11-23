#ifndef LOSS_FUNCTIONS_L1_H
#define LOSS_FUNCTIONS_L1_H

#include <cstdint>

namespace LossFunctions
{

struct L1Parameters
{
  uint32_t number_of_elements_;
  uint32_t stride_;
  // Also known as "dimensions", this is the number of grid "ticks" along a
  // single dimension.
  uint32_t dimension_length_;
};

//------------------------------------------------------------------------------
/// https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
/// Pointer alias is when 2 or more pointers access the same single memory
/// location.
//------------------------------------------------------------------------------
template <typename T>
__global__ void l1(
  const L1Parameters parameters,
  const float loss_scale,
  const T* __restrict__ predictions,
  const float* __restrict__ targets,
  float* __restrict__ values,
  T* __restrict__ gradients)
{
  const uint32_t i {threadIdx.x + blockIdx.x * blockDim.x};
  if (i >= parameters.number_of_elements_)
  {
    return;
  }

  const uint32_t inter_element_index {i / parameters.stride_};
  const uint32_t intra_element_index {i % parameters.stride_};

  if (intra_element_index >= parameters.dimension_length_)
  {
    values[i] = 0;
    gradients[i] = 0;
    return;
  }

  const uint32_t target_index {
    inter_element_index * parameters.dimension_length_ +
      intra_element_index};

  const float prediction {static_cast<float>(predictions[i])};

  const float pdf {1};

  const float difference {prediction - targets[target_index]};

  values[i] = fabsf(difference) / pdf;
}

} // namespace LossFunctions

#endif // LOSS_FUNCTIONS_L1_H
