#ifndef UTILITIES_DEVICE_MANAGEMENT_DEVICE_PROPERTIES_H
#define UTILITIES_DEVICE_MANAGEMENT_DEVICE_PROPERTIES_H

#include <cuda_runtime.h>
#include <string>
#include <string_view>

namespace Utilities
{
namespace DeviceManagement
{

class DeviceProperties
{
  public:
    DeviceProperties();
    DeviceProperties(const cudaDeviceProp device_properties);
    virtual ~DeviceProperties() = default;

    bool get_device_properties(const int device_id);

    //--------------------------------------------------------------------------
    /// @brief Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_13c26ab51c96f39b115d7826337541914
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7c4cb6c44a6c4608da36c44374499b31
    //--------------------------------------------------------------------------
    inline int get_supports_cooperative_kernels() const
    {
      return device_properties_.cooperativeLaunch;
    }

    //--------------------------------------------------------------------------
    /// @brief Maximum number of resident blocks per multiprocessor
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17f337476973ea65db85c277f4646f0b3
    //--------------------------------------------------------------------------
    inline int get_max_blocks_per_multiprocessor() const
    {
      return device_properties_.maxBlocksPerMultiProcessor;
    }

    //--------------------------------------------------------------------------
    /// @brief Maximum size of each dimension of a grid 
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17d138a572315b3bbb6caf7ccc914a130
    //--------------------------------------------------------------------------
    inline std::string get_max_grid_size() const
    {
      return format_int3(device_properties_.maxGridSize);
    }

    //--------------------------------------------------------------------------
    /// @brief ASCII string identifying device
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c
    //--------------------------------------------------------------------------
    inline std::string_view get_name() const
    {
      return device_properties_.name;
    }

    std::string get_compute_capability() const;

    //--------------------------------------------------------------------------
    /// @brief Global memory available on device in bytes
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1983c292e2078dd5a4240f49c41d647f3
    //--------------------------------------------------------------------------
    inline std::string get_global_memory_bytes() const
    {
      return format_to_bytes(device_properties_.totalGlobalMem);
    }

    static std::string format_to_bytes(const size_t bytes);

    static std::string format_int3(const int x[3]);

  private:
    cudaDeviceProp device_properties_;
};

} // namespace DeviceManagement
} // namespace Utilities

#endif // UTILITIES_DEVICE_MANAGEMENT_DEVICE_PROPERTIES_H