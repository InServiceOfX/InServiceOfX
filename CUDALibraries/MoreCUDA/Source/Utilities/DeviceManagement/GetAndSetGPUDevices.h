#ifndef UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_GPU_DEVICES_H
#define UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_GPU_DEVICES_H

namespace Utilities
{
namespace DeviceManagement
{

class GetAndSetGPUDevices
{
  public:
    GetAndSetGPUDevices();
    virtual ~GetAndSetGPUDevices() = default;

    //--------------------------------------------------------------------------
    /// @brief Returns number of compute-capable devices.
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f
    //--------------------------------------------------------------------------
    int get_device_count();

    //--------------------------------------------------------------------------
    /// @brief Get which device is currently being used.
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78
    /// @return True if successful, false otherwise.
    /// cudaSuccess, cudaErrorInvalidValue, cudaErrorDeviceUnavailable
    //--------------------------------------------------------------------------
    bool get_current_device();

    //--------------------------------------------------------------------------
    /// @brief Set device to be used for GPU execution.
    /// @ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb
    //--------------------------------------------------------------------------
    bool set_device(const int device_id);

    inline int device_count() const
    {
      return device_count_;
    }

    inline int current_device() const
    {
      return current_device_;
    }

  private:
    int device_count_;
    int current_device_;
};

} // namespace DeviceManagement
} // namespace Utilities

#endif // UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_GPU_DEVICES_H