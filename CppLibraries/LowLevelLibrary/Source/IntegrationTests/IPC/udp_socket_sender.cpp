#include "IPC/UDPSocket.h"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using IPC::UDPSocket;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;

namespace fs = std::filesystem;

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signal_handler(int signum)
{
  if (signum == SIGINT)
  {
    std::cout << 
      "\nReceived Ctrl+C. Once you enter a message to send, it'll "
        << "shut down...\n";
    g_running = false;
  }
}

int main()
{
  // Setup signal handler
  signal(SIGINT, signal_handler);

  // Get the path to configuration file
  fs::path current_file{__FILE__};
  fs::path test_data_dir{
      current_file.parent_path().parent_path() / "TestData"};
  
  // Create sender with configuration
  UDPSocketConfiguration transceiver_configuration {
    test_data_dir / "UDPTransceiverConfiguration.txt"};
  UDPSocketConfiguration sender_configuration {};
  sender_configuration.destination_ =
    transceiver_configuration.ip_address_and_port_;
  UDPSocket sender {sender_configuration};

  std::cout << "UDP Sender started. Type messages and press Enter to send.\n";
  std::cout << "Press Ctrl+C to exit.\n";

  std::string input {};
  while (g_running && std::getline(std::cin, input))
  {
    if (!input.empty())
    {
      std::vector<unsigned char> data {input.begin(), input.end()};
      sender.send(data);
      std::cout << "Sent: " << input << std::endl;
    }
  }

  return 0;
}
