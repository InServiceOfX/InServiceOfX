#ifndef UTILITIES_CONFIGURE_PATHS_H
#define UTILITIES_CONFIGURE_PATHS_H

#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;

namespace Utilities
{

class ConfigurePaths
{
  public:

    static constexpr std::string_view root_data_folder {"Data"};

    ConfigurePaths();

    fs::path project_path_;
    fs::path data_path_;
};

} // namespace Utilities

#endif // UTILITIES_CONFIGURE_PATHS_H