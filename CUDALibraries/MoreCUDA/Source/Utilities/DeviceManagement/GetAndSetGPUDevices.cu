#include "Utilities/DeviceManagement/GetAndSetGPUDevices.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>
#include <string_view>

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace Utilities
{
namespace DeviceManagement
{

GetAndSetGPUDevices::GetAndSetGPUDevices():
  device_count_{-1},
  current_device_{-1}
{
  get_device_count();
}

int GetAndSetGPUDevices::get_device_count()
{
  cudaGetDeviceCount(&device_count_);
  return device_count_;
}

bool GetAndSetGPUDevices::get_current_device()
{
  HandleUnsuccessfulCUDACall handle_get_current_device {
    "Failed to get current device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_get_current_device,
    cudaGetDevice(&current_device_));

  return handle_get_current_device.is_cuda_success();
}

bool GetAndSetGPUDevices::set_device(const int device_id)
{
  HandleUnsuccessfulCUDACall handle_set_device {
    "Failed to set device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_set_device,
    cudaSetDevice(device_id));

  return handle_set_device.is_cuda_success();
}

} // namespace DeviceManagement
} // namespace Utilities