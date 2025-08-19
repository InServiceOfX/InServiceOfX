#include "Utilities/DeviceManagement/DeviceProperties.h"
#include "Utilities/DeviceManagement/GetAndSetGPUDevices.h"

#include "gtest/gtest.h"

#include <iostream>

using ::Utilities::DeviceManagement::DeviceProperties;
using ::Utilities::DeviceManagement::GetAndSetGPUDevices;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace DeviceManagement
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DevicePropertiesTests, GetDevicePropertiesGetsDeviceProperties)
{
  static constexpr int device_id {0};

  GetAndSetGPUDevices gasgd {};
  ASSERT_TRUE(gasgd.set_device(device_id));

  DeviceProperties dp {};
  EXPECT_TRUE(dp.get_device_properties(device_id));

  // e.g. cooperative_launch_: 1
  std::cout << "cooperative_launch_: " <<
    dp.get_supports_cooperative_kernels() << std::endl;

  // e.g. max_blocks_per_multiprocessor_: 16
  std::cout << "max_blocks_per_multiprocessor_: " <<
    dp.get_max_blocks_per_multiprocessor() << std::endl;

  // e.g. max_grid_size_: [2147483647, 65535, 65535]
  std::cout << "max_grid_size_: " << dp.get_max_grid_size() << std::endl;

  // e.g. name_: NVIDIA GeForce RTX 3060
  std::cout << "name_: " << dp.get_name() << std::endl;

  // e.g. compute_capability_: 8.6
  std::cout << "compute_capability_: " << dp.get_compute_capability() <<
    std::endl;

  // e.g. global_memory_: 11.6 GB
  std::cout << "global_memory_: " << dp.get_global_memory_bytes() << std::endl;

  EXPECT_TRUE(true);
}

} // namespace DeviceManagement
} // namespace Utilities
} // namespace GoogleUnitTests