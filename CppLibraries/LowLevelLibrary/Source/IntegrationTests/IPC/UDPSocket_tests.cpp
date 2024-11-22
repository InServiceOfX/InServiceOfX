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

namespace IntegrationTests
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
fs::path test_data_dir {
  current_file.parent_path().parent_path().parent_path() / "UnitTests" / \
    "TestData"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(UDPSocketWithArbitraryIPAddressTest, SendsAndReceives)
{
  UDPSocketConfiguration receiver_configuration {};
  receiver_configuration.ip_address_and_port_ = Endpoint{"127.0.0.1", 10012};
  UDPSocket receiver {receiver_configuration};

  UDPSocketConfiguration transceiver_configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};
  UDPSocket transceiver {transceiver_configuration};

  UDPSocketConfiguration sender_configuration {};
  sender_configuration.destination_ = Endpoint{"172.16.1.100", 10011};
  UDPSocket sender {sender_configuration};

  const string message {"Hello, world!"};
  const vector<unsigned char> data {message.begin(), message.end()};
  vector<unsigned char> received_data {};

  sender.send(data);

  std::pair<std::size_t, Endpoint> result = transceiver.receive(received_data);
  EXPECT_EQ(result.first, 13);
  EXPECT_EQ(result.second.ip_address, "172.16.1.100");
  EXPECT_EQ(received_data, data);

  const string message_2 {"Goodbye, world!"};
  const vector<unsigned char> data_2 {message_2.begin(), message_2.end()};
  vector<unsigned char> received_data_2 {};

  transceiver.send(data_2);

  auto result2 = receiver.receive(received_data_2);
  EXPECT_EQ(result2.first, 15);
  EXPECT_EQ(result2.second.ip_address, "172.16.1.100");
  EXPECT_EQ(received_data_2, data_2);
}

} // namespace IPC

} // namespace IntegrationTests
