#include "IPC/UDP/RunningUDPTransceiver.h"
#include "UnitTests/ArbitraryIPAddressSetup.h"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>

using IPC::UDP::RunningUDPTransceiver;
using Testing::ArbitraryIPAddressSetup;
using UDPSocketConfiguration = IPC::UDPSocket::Configuration;

namespace fs = std::filesystem;

// Global pointer for signal handler to access
std::unique_ptr<ArbitraryIPAddressSetup> g_ip_setup {};

void signal_handler(int signum)
{
  if (signum == SIGINT)
  {
    std::cout << "\nReceived Ctrl+C. Cleaning up...\n";
    if (g_ip_setup)
    {
      g_ip_setup->tear_down();
    }
    exit(signum);
	}
}

int main()
{
  // Setup signal handler
  signal(SIGINT, signal_handler);

	// Setup arbitrary IP address
	g_ip_setup = std::make_unique<ArbitraryIPAddressSetup>();
	g_ip_setup->setup("172.16.1.100");

	// Get the path to configuration file
	fs::path current_file {__FILE__};
	fs::path test_data_dir{
		current_file.parent_path().parent_path().parent_path() / "TestData"};
	
	// Create and start transceiver
	UDPSocketConfiguration configuration {
		test_data_dir / "UDPTransceiverConfiguration.txt"};
	RunningUDPTransceiver transceiver {configuration};
	
	std::cout << "UDP Transceiver started. Press Ctrl+C to exit.\n";
	
	transceiver.start_listening();
	
	// Wait indefinitely
	while (true)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	
	return 0;
}
