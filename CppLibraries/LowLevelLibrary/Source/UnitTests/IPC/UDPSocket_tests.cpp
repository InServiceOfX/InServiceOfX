#include "IPC/UDPSocket.h"
#include "UnitTests/ArbitraryIPAddressSetup.h"

#include <filesystem> // std::filesystem::path
#include <gtest/gtest.h>
#include <string>
#include <vector>

using Endpoint = IPC::UDPSocket::Endpoint;
using IPC::UDPSocket;
using Testing::ArbitraryIPAddressSetup;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;
using std::string;
using std::vector;

namespace GoogleUnitTests
{

namespace IPC
{

class UDPSocketWithArbitraryIPAddressTest: public ::testing::Test
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

namespace fs = std::filesystem;

fs::path current_file {__FILE__};
fs::path test_data_dir {current_file.parent_path().parent_path() / "TestData"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(UDPSocketConfiguration, Constructs)
{
  UDPSocketConfiguration configuration {};
  configuration.ip_address_and_port_ = Endpoint{"127.0.0.1", 10011};
  EXPECT_EQ(configuration.ip_address_and_port_->ip_address, "127.0.0.1");
  EXPECT_EQ(configuration.ip_address_and_port_->port, 10011);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(UDPSocketConfiguration, ConfigurationParsesFromFilePath)
{
  UDPSocketConfiguration configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};
  EXPECT_EQ(configuration.ip_address_and_port_->ip_address, "172.16.1.100");
  EXPECT_EQ(configuration.ip_address_and_port_->port, 10011);
  EXPECT_EQ(configuration.destination_->ip_address, "127.0.0.1");
  EXPECT_EQ(configuration.destination_->port, 10012);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(UDPSocket, Constructs)
{
  {
    UDPSocketConfiguration configuration {};
    configuration.ip_address_and_port_ = Endpoint{"127.0.0.1", 10011};
    UDPSocket socket {configuration};
  }
  {
    UDPSocketConfiguration configuration {};
    configuration.destination_ = Endpoint{"127.0.0.1", 10012};
    UDPSocket socket {configuration};
  }

  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(UDPSocketWithArbitraryIPAddressTest, Constructs)
{
  UDPSocketConfiguration configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};
  UDPSocket socket {configuration};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(UDPSocket, SendsAndReceivesLocally)
{
  UDPSocketConfiguration receiver_configuration {};
  receiver_configuration.ip_address_and_port_ = Endpoint{"127.0.0.1", 12345};
  UDPSocket receiver {receiver_configuration};

  UDPSocketConfiguration sender_configuration {};
  sender_configuration.destination_ = Endpoint{"127.0.0.1", 12345};
  UDPSocket sender {sender_configuration};

  const string message {"Hello, world!"};
  const vector<unsigned char> data {message.begin(), message.end()};
  EXPECT_EQ(data.size(), 13);
  vector<unsigned char> received_data {};

  sender.send(data);
  std::pair<std::size_t, Endpoint> result = receiver.receive(received_data);
  EXPECT_EQ(result.first, 13);
  EXPECT_EQ(result.second.ip_address, "127.0.0.1");
  EXPECT_EQ(received_data, data);
}

} // namespace IPC

} // namespace GoogleUnitTests
