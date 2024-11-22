#ifndef IPC_UDPSOCKET_H
#define IPC_UDPSOCKET_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <netinet/in.h> // sockaddr_in
#include <optional>
#include <string>
#include <utility> // pair
#include <vector>

namespace IPC
{

class UDPSocket
{
  public:

    // Maximum UDP packet size from 16-bit length field in UDP header.
    static constexpr std::size_t DEFAULT_BUFFER_SIZE {65507};

    struct Endpoint
    {
      std::string ip_address;
      uint16_t port;
    };

    struct Configuration
    {
      std::optional<Endpoint> ip_address_and_port_;
      std::optional<std::chrono::milliseconds> receive_timeout_;
      std::optional<Endpoint> destination_;

      Configuration(
        const std::optional<Endpoint>& ip_address_and_port=std::nullopt,
        const std::optional<std::chrono::milliseconds>
          receive_timeout=std::nullopt,
        const std::optional<Endpoint>& destination=std::nullopt
      ):
        ip_address_and_port_{ip_address_and_port},
        receive_timeout_{receive_timeout},
        destination_{destination}
      {};

      Configuration(const std::filesystem::path& configuration_file_path);
    };

    static sockaddr_in create_address(const Endpoint& endpoint);

    UDPSocket(const Configuration& configuration);

    virtual ~UDPSocket();

    // Delete copy operations, allow move
    UDPSocket(const UDPSocket&) = delete;
    UDPSocket& operator=(const UDPSocket&) = delete;
    UDPSocket(UDPSocket&&) noexcept;
    UDPSocket& operator=(UDPSocket&&) noexcept;

    std::size_t send_to(
      const std::vector<unsigned char>& data,
      const Endpoint& destination);

    std::size_t send(const std::vector<unsigned char>& data);

    std::pair<std::size_t, Endpoint> receive(std::vector<unsigned char>& data);

  protected:

    void set_receive_timeout(const std::chrono::milliseconds& receive_timeout);

  private:

    int socket_fd_;
    std::optional<Endpoint> destination_;
};

} // namespace IPC

#endif // IPC_UDPSOCKET_H