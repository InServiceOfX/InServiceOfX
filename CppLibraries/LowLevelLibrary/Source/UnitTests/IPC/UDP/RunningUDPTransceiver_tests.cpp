#include "IPC/UDP/RunningUDPTransceiver.h"
#include "UnitTests/ArbitraryIPAddressSetup.h"

#include <gtest/gtest.h>

using IPC::UDP::RunningUDPTransceiver;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;
using Testing::ArbitraryIPAddressSetup;

namespace GoogleUnitTests
{

namespace IPC
{

namespace UDP
{

namespace fs = std::filesystem;

fs::path current_file {__FILE__};
fs::path test_data_dir {
  current_file.parent_path().parent_path().parent_path() / "TestData"};

class RunningUDPTransceiverTest: public ::testing::Test
{
  public:

    ArbitraryIPAddressSetup arbitrary_ip_address_setup_;

  protected:

    void SetUp() override
    {
      arbitrary_ip_address_setup_.setup("172.16.1.100");
    }

    void TearDown() override
    {
      arbitrary_ip_address_setup_.tear_down();
    }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(RunningUDPTransceiverTest, Constructs)
{
  RunningUDPTransceiver transceiver {
    UDPSocketConfiguration{test_data_dir / "UDPTransceiverConfiguration.txt"}};

  EXPECT_TRUE(true);
}

} // namespace UDP

} // namespace IPC

} // namespace GoogleUnitTests
