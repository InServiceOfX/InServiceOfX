#ifndef UTILITIES_PARSERS_PARSE_SIMPLE_MAP_H
#define UTILITIES_PARSERS_PARSE_SIMPLE_MAP_H

#include <filesystem>
#include <string>
#include <unordered_map>

namespace Utilities
{
namespace Parsers
{

std::unordered_map<std::string, std::string> parse_simple_map(
    const std::filesystem::path& filename);

} // namespace Parsers
} // namespace Utilities

#endif // UTILITIES_PARSERS_PARSE_SIMPLE_MAP_H