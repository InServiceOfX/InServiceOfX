#include "parse_simple_map.h"

#include <algorithm> // std::find_if
#include <cctype> // std::isspace
#include <filesystem> // std::filesystem::path
#include <fstream> // std::ifstream
#include <stdexcept>
#include <string> // std::getline

using std::find_if;

namespace Utilities
{
namespace Parsers
{

std::unordered_map<std::string, std::string> parse_simple_map(
    const std::filesystem::path& filename)
{
  std::unordered_map<std::string, std::string> result {};
  std::ifstream file{filename};

  if (!file)
  {
    throw std::runtime_error("Failed to open file: " + filename.string());
  }

  std::string line {};
  while (std::getline(file, line))
  {
    // Remove comments
    auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos)
    {
      line.erase(comment_pos);
    }

    // Trim leading and trailing whitespace
    auto trim = [](std::string& str)
    {
      auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };

      // Trim left
      str.erase(str.begin(), find_if(str.begin(), str.end(), is_not_space));
      // Trim right
      str.erase(
        find_if(
          str.rbegin(),
          str.rend(),
          is_not_space).base(),
        str.end());
    };
    trim(line);

    // Skip empty lines
    if (line.empty()) {
        continue;
    }

    // Split into key and value
    auto delimiter_pos = line.find(':');
    if (delimiter_pos == std::string::npos)
    {
      continue; // No delimiter found; skip this line
    }

    std::string key {line.substr(0, delimiter_pos)};
    std::string value {line.substr(delimiter_pos + 1)};

    trim(key);
    trim(value);

    result.emplace(std::move(key), std::move(value));
  }

  return result;
}

} // namespace Parsers
} // namespace Utilities