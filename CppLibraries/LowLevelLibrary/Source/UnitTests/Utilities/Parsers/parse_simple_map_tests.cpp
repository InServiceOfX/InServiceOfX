#include "Utilities/Parsers/parse_simple_map.h"
#include <filesystem> // std::filesystem::path
#include <gtest/gtest.h>

using Utilities::Parsers::parse_simple_map;
namespace fs = std::filesystem;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace Parsers
{

fs::path current_file {__FILE__};
fs::path test_data_dir {
  current_file.parent_path().parent_path().parent_path() / "TestData"};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
TEST(parse_simple_map, Parses)
{
  auto result = parse_simple_map(
    test_data_dir / "UDPTransceiverConfiguration.txt");
  EXPECT_EQ(result.size(), 4);
  EXPECT_EQ(result["ip_address"], "172.16.1.100");
  EXPECT_EQ(result["port"], "10011");
  EXPECT_EQ(result["destination_ip_address"], "127.0.0.1");
  EXPECT_EQ(result["destination_port"], "10012");
}

} // namespace Parsers
} // namespace Utilities
} // namespace GoogleUnitTests
