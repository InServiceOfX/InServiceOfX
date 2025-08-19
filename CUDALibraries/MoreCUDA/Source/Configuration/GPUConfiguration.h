#ifndef CONFIGURATION_GPU_CONFIGURATION_H
#define CONFIGURATION_GPU_CONFIGURATION_H

#include "Configuration/YAMLKeyValueConfiguration.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Configuration
{

struct GPUConfigurationData
{
  int device_id {-1};
};

class GPUConfiguration : public YAMLKeyValueConfiguration<GPUConfigurationData>
{
  public:

    GPUConfiguration() = default;
    
  protected:
    virtual std::vector<FieldDefinition> define_fields() const override;
    virtual void update_configuration_from_values(
      const std::unordered_map<std::string, ConfigurationValue>& values)
        override;
};

} // namespace Configuration

#endif // CONFIGURATION_GPU_CONFIGURATION_H
