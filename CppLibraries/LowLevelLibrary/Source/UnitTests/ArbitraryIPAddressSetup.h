#ifndef UNIT_TESTS_ARBITRARY_IP_ADDRESS_SETUP_H
#define UNIT_TESTS_ARBITRARY_IP_ADDRESS_SETUP_H

#include <string>

namespace Testing
{

class ArbitraryIPAddressSetup
{
  public:

    ArbitraryIPAddressSetup(const std::string& ip_address=""):
      ip_address_{ip_address},
      // Default to eth0
      interface_name_{"eth0"}
    {}

    void setup(const std::string& ip_address);

    void tear_down();

    ~ArbitraryIPAddressSetup() = default;

    // Prevent copying
    ArbitraryIPAddressSetup(const ArbitraryIPAddressSetup&) = delete;
    ArbitraryIPAddressSetup& operator=(const ArbitraryIPAddressSetup&) = delete;

  private:
    std::string ip_address_;
    std::string interface_name_;
};

} // namespace Testing

#endif // UNIT_TESTS_ARBITRARY_IP_ADDRESS_SETUP_H