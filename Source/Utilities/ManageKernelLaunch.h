#ifndef UTILITIES_MANAGE_KERNEL_LAUNCH_H
#define UTILITIES_MANAGE_KERNEL_LAUNCH_H

#include <cstdint>
#include <cudnn.h>

namespace Utilities
{

template <typename T>
__host__ __device__ divide_and_round_up(T value, T divisor)
{
  return (value + divisor - 1) / divisor;
}

struct DefaultKernelLaunchValues
{
  static constexpr uint32_t batch_size_granularity_ {256};
  static constexpr uint32_t number_of_threads_1D_ {128};
  static constexpr uint32_t warp_size_ {32};
};

template <typename T>
constexpr __host__ __device__ uint32_t get_number_of_blocks_1D(
  T number_of_elements,
  const uint32_t number_of_threads =
    DefaultKernelLaunchValues::number_of_threads_1D_)
{
  return static_cast<uint32_t>(
    divide_and_round_up(number_of_elements, static_cast<T>(number_of_threads)));
}

} // namespace Utilities

#endif // UTILITIES_MANAGE_KERNEL_LAUNCH_H