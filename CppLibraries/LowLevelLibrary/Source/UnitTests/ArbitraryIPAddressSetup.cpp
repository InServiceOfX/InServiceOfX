#include "ArbitraryIPAddressSetup.h"

#include <arpa/inet.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace Testing
{

void ArbitraryIPAddressSetup::setup(const std::string& ip_address)
{
  if (!ip_address.empty())
  {
    ip_address_ = ip_address;
  }

  // Find active ethernet interface
  FILE* fp = popen(
    "ip -o link show | awk '/state UP/ {print $2}' | head -n1", "r");
  if (!fp)
  {
    throw std::runtime_error("Failed to find network interface");
  }

  char buffer[128];
  if (fgets(buffer, sizeof(buffer), fp) != nullptr)
  {
    interface_name_ = std::string {buffer};
    interface_name_ = interface_name_.substr(0, interface_name_.find_first_of(":\n"));
  }
  pclose(fp);

  // Try to add IP address to interface, ignore if it already exists
  std::string cmd {"sudo ip addr add " + ip_address_ + "/24 dev " + interface_name_ + " 2>&1"};
  FILE* result {popen(cmd.c_str(), "r")};
  if (result)
  {
    char error_buffer[256];
    if (fgets(error_buffer, sizeof(error_buffer), result) != nullptr)
    {
      std::string error_msg {error_buffer};
      // If error is not "File exists", then it's a real error
      if (error_msg.find("File exists") == std::string::npos)
      {
        pclose(result);
        throw std::runtime_error("Failed to add IP address: " + error_msg);
      }
    }
    pclose(result);
  }
}

void ArbitraryIPAddressSetup::tear_down()
{
  // Remove IP address from interface
  std::string command {
    "sudo ip addr del " + ip_address_ + "/24 dev " + interface_name_};
  system(command.c_str());
}

} // namespace Testing