#include "IPC/UDPSocket.h"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using IPC::UDPSocket;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;
using std::string;

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
    std::cout << 
      "\nReceived Ctrl+C. Once you enter a message to send, it'll "
        << "shut down...\n";
    g_running = false;
  }
}

int main(int argc, char* argv[])
{
  if (argc > 1 && (string(argv[1]) == "-h" || string(argv[1]) == "--help"))
  {
    print_usage(argv[0]);
    return 0;
  }

  // Setup signal handler
  signal(SIGINT, signal_handler);

  // Get the path to configuration file
  fs::path current_file{__FILE__};
  fs::path test_data_dir{
    current_file.parent_path().parent_path() / "TestData"};
    
  string config_file = "UDPTransceiverConfiguration.txt";
  if (argc > 1)
  {
    config_file = argv[1];
  }
    
  // Create sender with configuration
  UDPSocketConfiguration transceiver_configuration{
    test_data_dir / config_file};
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
