#ifndef CONFIGURATION_YAML_KEY_VALUE_CONFIGURATION_H
#define CONFIGURATION_YAML_KEY_VALUE_CONFIGURATION_H

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <iostream>

namespace Configuration
{

// Supported value types
using ConfigurationValue = std::variant<int, double, bool, std::string>;

struct FieldDefinition
{
  std::string name_;
  std::function<bool(const std::string&, ConfigurationValue&)> parser_;
  ConfigurationValue default_value_;
  std::string description_;

  FieldDefinition(
    const std::string& field_name,
    std::function<bool(const std::string&, ConfigurationValue&)> parse_function,
    const ConfigurationValue& default_value,
    const std::string& description = ""
  ):
  name_{field_name},
  parser_{parse_function},
  default_value_{default_value},
  description_{description}
  {}
};

//------------------------------------------------------------------------------
/// @brief This is a "base class" specifying for a text file meant for
/// configuration and is of the format of key-value pairs that is compatible to
/// YAML.
/// @tparam ConfigurationStruct This has the "field names" to parse for.
//------------------------------------------------------------------------------
template<typename ConfigurationStruct>
class YAMLKeyValueConfiguration
{
  public:
    YAMLKeyValueConfiguration() = default;

    bool parse_configuration_file(const std::string& file_path);
    bool parse_configuration_file(const std::filesystem::path& file_path);

    inline ConfigurationStruct get_configuration_struct() const
    {
      return configuration_struct_;
    }

  protected:
    virtual std::vector<FieldDefinition> define_fields() const = 0;
    virtual void update_configuration_from_values(
      const std::unordered_map<std::string, ConfigurationValue>& values) = 0;

    //--------------------------------------------------------------------------
    /// Built-in parser implementations
    //--------------------------------------------------------------------------
    static bool parse_int(const std::string& value, ConfigurationValue& result);
    static bool parse_double(
      const std::string& value,
      ConfigurationValue& result);
    static bool parse_bool(
      const std::string& value,
      ConfigurationValue& result);
    inline static bool parse_string(
      const std::string& value,
      ConfigurationValue& result)
    {
      result = value;
      return true;
    }

    ConfigurationStruct configuration_struct_;

  private:
    bool parse_line(
      const std::string& line,
      std::unordered_map<std::string, ConfigurationValue>& values);

    std::optional<std::pair<std::string, std::string>> parse_key_value(
      const std::string& line);

    std::string trim_whitespace(const std::string& str);
    bool is_comment_line(const std::string& line);
};

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_configuration_file(
  const std::string& file_path)
{
  return parse_configuration_file(std::filesystem::path(file_path));
}

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_configuration_file(
  const std::filesystem::path& file_path)
{
  std::ifstream input_file {file_path};
  if (!input_file.is_open())
  {
    std::cerr << "Error: Could not open configuration file: " << file_path
      << std::endl;
    return false;
  }

  std::string line {};
  int line_number {0};
  std::unordered_map<std::string, ConfigurationValue> parsed_values {};

  // Initialize with default values
  for (const auto& field : define_fields())
  {
    parsed_values[field.name_] = field.default_value_;
  }

  while (std::getline(input_file, line))
  {
    line_number++;
    
    if (line.empty() || is_comment_line(line))
    {
      continue;
    }
    
    if (!parse_line(line, parsed_values))
    {
      std::cerr << "Warning: Could not parse line " << line_number << " in " <<
        file_path << ": " << line << std::endl;
      continue;
    }
  }

  update_configuration_from_values(parsed_values);
    
  std::cout << "Successfully parsed configuration from: " << file_path <<
    std::endl;
  return true;
}

//--------------------------------------------------------------------------
/// Built-in parser implementations
//--------------------------------------------------------------------------

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_int(
  const std::string& value,
  ConfigurationValue& result)
{
  try
  {
    result = std::stoi(value);
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_double(
  const std::string& value,
  ConfigurationValue& result)
{
  try
  {
    result = std::stod(value);
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_bool(
  const std::string& value,
  ConfigurationValue& result)
{
  std::string lower_value {value};
  std::transform(
    lower_value.begin(),
    lower_value.end(),
    lower_value.begin(),
    ::tolower);
  
  if (lower_value == "true" || lower_value == "1" || lower_value == "yes")
  {
    result = true;
    return true;
  }
  else if (
    lower_value == "false" || 
      lower_value == "0" ||
      lower_value == "no")
  {
      result = false;
      return true;
  }
  return false;
}

//--------------------------------------------------------------------------
/// Helper functions for parsing lines.
//--------------------------------------------------------------------------

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::parse_line(
  const std::string& line,
  std::unordered_map<std::string, ConfigurationValue>& values)
{
  auto key_value = parse_key_value(line);
  if (!key_value)
  {
    return false;
  }

  const auto& [key, value] = *key_value;

  // Find the field definition
  for (const auto& field : define_fields())
  {
    if (field.name_ == key)
    {
      ConfigurationValue parsed_value {};
      if (field.parser_(value, parsed_value))
      {
        values[key] = parsed_value;
        return true;
      }
      else
      {
        std::cerr << "Error parsing value '" << value << "' for field '" <<
          key << "'" << std::endl;
        return false;
      }
    }
  }

  std::cerr << "Warning: Unknown configuration key: " << key << std::endl;
  return false;
}
     
template<typename ConfigurationStruct>
std::optional<std::pair<std::string, std::string>>
  YAMLKeyValueConfiguration<ConfigurationStruct>::parse_key_value(
    const std::string& line)
{
  std::string trimmed {trim_whitespace(line)};
  
  // Find the colon separator
  const size_t colon_pos {trimmed.find(':')};
  if (colon_pos == std::string::npos)
  {
    return std::nullopt;
  }
  
  // Extract key and value
  std::string key {trim_whitespace(trimmed.substr(0, colon_pos))};
  std::string value {trim_whitespace(trimmed.substr(colon_pos + 1))};
  
  // Validate key and value
  if (key.empty() || value.empty())
  {
    return std::nullopt;
  }
  
  return std::make_pair(key, value);
}

template<typename ConfigurationStruct>
std::string YAMLKeyValueConfiguration<ConfigurationStruct>::trim_whitespace(
  const std::string& str)
{
  auto start = str.begin();
  auto end = str.end();
  
  // Trim from start
  while (start != end && std::isspace(*start))
  {
    start++;
  }
  
  // Trim from end
  while (start != end && std::isspace(*(end - 1)))
  {
    end--;
  }
  
  return std::string(start, end);
}

template<typename ConfigurationStruct>
bool YAMLKeyValueConfiguration<ConfigurationStruct>::is_comment_line(
  const std::string& line)
{
  const std::string trimmed {trim_whitespace(line)};
  return trimmed.empty() || trimmed[0] == '#';
}



} // namespace Configuration

#endif // CONFIGURATION_YAML_KEY_VALUE_CONFIGURATION_H
