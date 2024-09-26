#include "ConfigurePaths.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace Utilities
{

ConfigurePaths::ConfigurePaths()
{
  // https://en.cppreference.com/w/cpp/preprocessor/replace
  // Expands to name of the current file, as a character string literal, and is
  // changed by # line directive.      
  // https://en.cppreference.com/w/cpp/filesystem/absolute
  // std::filesystem::absolute returns path referencing same file system
  // location as the path.
  fs::path source_file_path {fs::absolute(__FILE__)};

  project_path_ = source_file_path.parent_path().parent_path().parent_path()
    .parent_path();

  data_path_ = project_path_ / root_data_folder;
}

} // namespace Utilities