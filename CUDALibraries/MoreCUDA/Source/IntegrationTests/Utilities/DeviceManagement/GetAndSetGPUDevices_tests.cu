#include "Configuration/GPUConfiguration.h"
#include "Utilities/DeviceManagement/GetAndSetGPUDevices.h"
#include "gtest/gtest.h"

#include <iostream>

using ::Configuration::GPUConfiguration;
using ::Utilities::DeviceManagement::GetAndSetGPUDevices;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace DeviceManagement
{

std::filesystem::path get_test_file_path()
{
  return std::filesystem::path(__FILE__).parent_path() /
    "../../../../Configurations/GPUConfiguration.txt";
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetAndSetGPUDevicesTests, GetDeviceCountGetsDeviceCount)
{
  GetAndSetGPUDevices gasgd {};

  std::cout << "device_count_: " << gasgd.get_device_count() << std::endl;
  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetAndSetGPUDevicesTests, GetCurrentDeviceGetsCurrentDevice)
{
  GetAndSetGPUDevices gasgd {};
  EXPECT_TRUE(gasgd.get_current_device());

  std::cout << "current_device_: " << gasgd.current_device() << std::endl;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetAndSetGPUDevicesTests, SetDeviceFromConfiguration)
{
  GPUConfiguration gpu_configuration {};
  ASSERT_TRUE(std::filesystem::exists(get_test_file_path()));

  gpu_configuration.parse_configuration_file(get_test_file_path());

  GetAndSetGPUDevices gasgd {};
  EXPECT_TRUE(gasgd.set_device(
    gpu_configuration.get_configuration_struct().device_id
  ));
  EXPECT_TRUE(gasgd.get_current_device());

  std::cout << "current_device_: " << gasgd.current_device() << std::endl;
}

} // namespace DeviceManagement
} // namespace Utilities
} // namespace GoogleUnitTests