#include "Configuration/GPUConfiguration.h"
#include "gtest/gtest.h"

#include <filesystem>

using ::Configuration::GPUConfiguration;

namespace GoogleUnitTests
{
namespace Configuration
{

std::filesystem::path get_test_file_path()
{
  return std::filesystem::path(__FILE__).parent_path() /
    "../../../Configurations/GPUConfiguration.txt";
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GPUConfigurationTests, GPUConfigurationGetsGPUConfiguration)
{
  GPUConfiguration gpu_configuration {};
  EXPECT_EQ(gpu_configuration.get_configuration_struct().device_id, -1);

  ASSERT_TRUE(std::filesystem::exists(get_test_file_path()));

  gpu_configuration.parse_configuration_file(get_test_file_path());
  EXPECT_EQ(gpu_configuration.get_configuration_struct().device_id, 0);
}

} // namespace Configuration
} // namespace GoogleUnitTests