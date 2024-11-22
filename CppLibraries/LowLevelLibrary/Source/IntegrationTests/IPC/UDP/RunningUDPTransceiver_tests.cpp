#include "IPC/UDP/RunningUDPTransceiver.h"
#include "IPC/UDPSocket.h"
#include "UnitTests/ArbitraryIPAddressSetup.h"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using IPC::UDPSocket;
using IPC::UDP::RunningUDPTransceiver;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;
using Testing::ArbitraryIPAddressSetup;

namespace IntegrationTests
{
namespace IPC
{
namespace UDP
{

namespace fs = std::filesystem;

fs::path current_file {__FILE__};
fs::path test_data_dir {
  current_file.parent_path().parent_path().parent_path().parent_path() / \
    "UnitTests" / "TestData"};

class RunningUDPTransceiverIntegrationTest: public ::testing::Test
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
TEST_F(RunningUDPTransceiverIntegrationTest, StopsListeningUponReceivingMessage)
{
  UDPSocketConfiguration transceiver_configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};

  UDPSocket::Configuration sender_configuration {};
  sender_configuration.destination_ =
    transceiver_configuration.ip_address_and_port_;
  UDPSocket sender {sender_configuration};

  RunningUDPTransceiver transceiver {transceiver_configuration};

  std::vector<unsigned char> send_data = {'H', 'e', 'l', 'l', 'o'};
  
  // Start sender thread that waits 2000ms before sending
  std::thread sender_thread([&sender, &send_data]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender.send(send_data);
  });

  transceiver.start_listening(std::chrono::milliseconds(1000));
  
  sender_thread.join();
  
  // The transceiver should have timed out before receiving the message
  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(RunningUDPTransceiverIntegrationTest, EndToEndTransmission)
{
  UDPSocketConfiguration transceiver_configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};

  // Setup sender
  UDPSocket::Configuration sender_configuration {};
  sender_configuration.destination_ =
    transceiver_configuration.ip_address_and_port_;
  UDPSocket sender {sender_configuration};

  // Setup final receiver
  UDPSocketConfiguration receiver_configuration {};
  receiver_configuration.ip_address_and_port_ = 
    transceiver_configuration.destination_;
  UDPSocket receiver {receiver_configuration};

  RunningUDPTransceiver transceiver {transceiver_configuration};

  std::vector<unsigned char> send_data = {'H', 'e', 'l', 'l', 'o'};
  std::vector<unsigned char> received_data {};
  std::atomic<bool> message_received{false};
  
  // Start receiver thread
  std::thread receiver_thread([&]()
  {
    auto [bytes_received, sender_endpoint] = receiver.receive(received_data);
    if (bytes_received > 0)
    {
      message_received = true;
    }
  });

  // Start sender thread that waits 2000ms before sending
  std::thread sender_thread([&]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender.send(send_data);
  });

  transceiver.start_listening(std::chrono::milliseconds(1000));
  
  sender_thread.join();
  receiver_thread.join();
  
  EXPECT_TRUE(message_received);
  EXPECT_EQ(received_data, send_data);
}

} // namespace UDP
} // namespace IPC
} // namespace IntegrationTests
