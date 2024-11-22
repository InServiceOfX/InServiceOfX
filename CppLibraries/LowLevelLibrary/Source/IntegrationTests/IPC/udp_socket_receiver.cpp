#include "IPC/UDPSocket.h"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using IPC::UDPSocket;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;

namespace fs = std::filesystem;

void print_usage(const char* program_name)
{
  std::cout << "Usage: " << program_name << " [configuration_file]\n"
            << "  configuration_file: Optional. Path to configuration file in TestData directory\n"
            << "                     (defaults to UDPTransceiverConfiguration.txt)\n"
            << "  -h, --help: Show this help message\n";
}

// Global flag for clean shutdown
std::atomic<bool> g_running{true};

void signal_handler(int signum)
{
  if (signum == SIGINT)
  {
    std::cout << "\nReceived Ctrl+C. Shutting down...\n";
    g_running = false;
  }
}

int main(int argc, char* argv[])
{
  if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
  {
    print_usage(argv[0]);
    return 0;
  }

  // Setup signal handler
  signal(SIGINT, signal_handler);

  // Get the path to configuration file
  fs::path current_file {__FILE__};
  fs::path test_data_dir {
    current_file.parent_path().parent_path() / "TestData"};
    
  std::string config_file = "UDPTransceiverConfiguration.txt";
  if (argc > 1)
  {
    config_file = argv[1];
  }
    
  // Create receiver with configuration from transceiver's destination
  UDPSocketConfiguration transceiver_configuration {
    test_data_dir / config_file};
  UDPSocketConfiguration receiver_configuration {};
  receiver_configuration.ip_address_and_port_ = 
    transceiver_configuration.destination_;
  UDPSocket receiver {receiver_configuration};

  std::cout << "UDP Receiver started. Listening for messages...\n";
  std::cout << std::left << std::setw(20) << "Message Length" 
    << "Message Content\n";
  std::cout << std::string(50, '-') << "\n";

  std::vector<unsigned char> data {};
  while (g_running)
  {
    auto [bytes_received, sender_endpoint] = receiver.receive(data);
    if (bytes_received > 0)
    {
      std::string message(data.begin(), data.end());
      std::cout << std::left << std::setw(20) << bytes_received 
        << message << "\n";
    }
  }

  return 0;
}