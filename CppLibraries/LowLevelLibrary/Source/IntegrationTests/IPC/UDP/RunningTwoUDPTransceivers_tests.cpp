#include "IPC/UDP/RunningTwoUDPTransceivers.h"
#include "IPC/UDPSocket.h"
#include "UnitTests/ArbitraryIPAddressSetup.h"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using IPC::UDPSocket;
using IPC::UDP::RunningTwoUDPTransceivers;
using Testing::ArbitraryIPAddressSetup;

namespace IntegrationTests
{
namespace IPC
{
namespace UDP
{

namespace fs = std::filesystem;

fs::path running_two_udp_transceivers_current_file {__FILE__};
fs::path running_two_udp_transceivers_test_data_dir {
  running_two_udp_transceivers_current_file.parent_path().parent_path()
    .parent_path() / "TestData"};

class RunningTwoUDPTransceiversIntegrationTest: public ::testing::Test
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
TEST_F(
  RunningTwoUDPTransceiversIntegrationTest,
  StopsListeningUponReceivingMessages)
{
  fs::path test_data_dir {running_two_udp_transceivers_test_data_dir};
  RunningTwoUDPTransceivers::Configuration configuration {
    test_data_dir / "RunningTwoUDPTransceiversConfiguration.txt"};

  // Setup senders
  UDPSocket::Configuration sender1_configuration{};
  sender1_configuration.destination_ = configuration.ip_address_and_port_1;
  UDPSocket sender1{sender1_configuration};

  UDPSocket::Configuration sender2_configuration{};
  sender2_configuration.destination_ = configuration.ip_address_and_port_2;
  UDPSocket sender2{sender2_configuration};

  RunningTwoUDPTransceivers transceivers {configuration};

  std::vector<unsigned char> send_data1 {'H', 'e', 'l', 'l', 'o', '1'};
  std::vector<unsigned char> send_data2 {'H', 'e', 'l', 'l', 'o', '2'};

  // Start sender threads that wait before sending
  std::thread sender1_thread([&]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender1.send(send_data1);
  });

  std::thread sender2_thread([&]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender2.send(send_data2);
  });

  transceivers.start_listening(std::chrono::milliseconds(1000));

  sender1_thread.join();
  sender2_thread.join();

  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST_F(RunningTwoUDPTransceiversIntegrationTest, EndToEndTransmission)
{
  fs::path test_data_dir {running_two_udp_transceivers_test_data_dir};

  RunningTwoUDPTransceivers::Configuration configuration{
    test_data_dir / "RunningTwoUDPTransceiversConfiguration.txt"};

  // Setup senders
  UDPSocket::Configuration sender1_configuration{};
  sender1_configuration.destination_ = configuration.ip_address_and_port_1;
  UDPSocket sender1{sender1_configuration};

  UDPSocket::Configuration sender2_configuration{};
  sender2_configuration.destination_ = configuration.ip_address_and_port_2;
  UDPSocket sender2{sender2_configuration};

  // Setup final receiver
  UDPSocket::Configuration receiver_configuration{};
  receiver_configuration.ip_address_and_port_ = configuration.destination_1;
  UDPSocket receiver{receiver_configuration};

  RunningTwoUDPTransceivers transceivers{configuration};

  std::vector<unsigned char> send_data1 = {'H', 'e', 'l', 'l', 'o', '1'};
  std::vector<unsigned char> send_data2 = {'H', 'e', 'l', 'l', 'o', '2'};
  std::vector<unsigned char> received_data {};
  std::atomic<int> messages_received {0};
  
  // Start receiver thread
  std::thread receiver_thread([&]()
  {
    while (messages_received < 2)
    {
      auto [bytes_received, sender_endpoint] = receiver.receive(received_data);
      if (bytes_received > 0)
      {
        messages_received++;
      }
    }
  });

  // Start sender threads that wait before sending
  std::thread sender1_thread([&]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender1.send(send_data1);
  });

  std::thread sender2_thread([&]()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    sender2.send(send_data2);
  });

  transceivers.start_listening(std::chrono::milliseconds(1000));
  
  sender1_thread.join();
  sender2_thread.join();
  receiver_thread.join();
  
  EXPECT_EQ(messages_received, 2);
}

} // namespace UDP
} // namespace IPC
} // namespace IntegrationTests
