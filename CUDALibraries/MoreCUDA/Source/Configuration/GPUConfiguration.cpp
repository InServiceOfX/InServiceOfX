#include "Configuration/GPUConfiguration.h"
#include "Configuration/YAMLKeyValueConfiguration.h"

#include <string>
#include <unordered_map>
#include <utility> // std::get

namespace Configuration
{

std::vector<FieldDefinition> GPUConfiguration::define_fields() const
{
  return {
    FieldDefinition(
      "device_id",
      &YAMLKeyValueConfiguration::parse_int,
      -1,
      "The ID of the GPU to use"),
  };
}

void GPUConfiguration::update_configuration_from_values(
  const std::unordered_map<std::string, ConfigurationValue>& values)
{
  for (const auto& [key, value] : values)
  {
    if (key == "device_id")
    {
      configuration_struct_.device_id = std::get<int>(value);
    }
  }
}

} // namespace Configuration
