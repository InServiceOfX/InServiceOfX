#include "IPC/UDPSocket.h"
#include "Utilities/Parsers/parse_simple_map.h"

#include <arpa/inet.h> // inet_addr
#include <chrono>
#include <filesystem>
#include <netinet/in.h> // sockaddr_in
#include <optional>
#include <stdexcept>
#include <string> // std::stoul
#include <sys/socket.h>
#include <unistd.h>
#include <utility> // pair
#include <vector>

using Utilities::Parsers::parse_simple_map;

namespace IPC
{

UDPSocket::Configuration::Configuration(
  const std::filesystem::path& configuration_file_path)
{
  const auto configuration_map = parse_simple_map(configuration_file_path);

  auto iterator_ip_address = configuration_map.find("ip_address");
  auto iterator_port = configuration_map.find("port");

  if (iterator_ip_address == configuration_map.end() ||
      iterator_port == configuration_map.end())
  {
    ip_address_and_port_ = std::nullopt;
  }
  else
  {
    ip_address_and_port_ = Endpoint{
      iterator_ip_address->second,
      static_cast<uint16_t>(std::stoul(iterator_port->second))
    };
  }

  auto iterator_receive_timeout = configuration_map.find("receive_timeout");
  if (iterator_receive_timeout == configuration_map.end())
  {
    receive_timeout_ = std::nullopt;
  }
  else
  {
    receive_timeout_ = std::chrono::milliseconds(
      std::stoul(iterator_receive_timeout->second));
  }

  auto iterator_destination_ip_address = configuration_map.find(
    "destination_ip_address");
  auto iterator_destination_port = configuration_map.find("destination_port");

  if (iterator_destination_ip_address == configuration_map.end() ||
      iterator_destination_port == configuration_map.end())
  {
    destination_ = std::nullopt;
  }
  else
  {
    destination_ = Endpoint{
      iterator_destination_ip_address->second,
      static_cast<uint16_t>(std::stoul(iterator_destination_port->second))
    };
  }
}

sockaddr_in UDPSocket::create_address(const Endpoint& endpoint)
{
  sockaddr_in address {};
  address.sin_family = AF_INET;
  address.sin_port = htons(endpoint.port);
  address.sin_addr.s_addr = inet_addr(endpoint.ip_address.c_str());
  return address;
}

UDPSocket::UDPSocket(const Configuration& configuration):
  socket_fd_{-1},
  destination_{configuration.destination_}
{
  socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ == -1)
  {
    throw std::runtime_error("Failed to create socket");
  }

  if (configuration.ip_address_and_port_)
  {
    auto address = create_address(
      configuration.ip_address_and_port_.value());
    if (bind(
      socket_fd_,
      reinterpret_cast<sockaddr*>(&address),
      sizeof(address)) == -1)
    {
      close(socket_fd_);
      throw std::runtime_error("Failed to bind socket");
    }
  }

  if (configuration.receive_timeout_)
  {
    set_receive_timeout(configuration.receive_timeout_.value());
  }
}

UDPSocket::~UDPSocket()
{
  close(socket_fd_);
}

UDPSocket::UDPSocket(UDPSocket&& other) noexcept:
  socket_fd_{other.socket_fd_}
{
  other.socket_fd_ = -1;
}

UDPSocket& UDPSocket::operator=(UDPSocket&& other) noexcept
{
  if (this != &other)
  {
    if (socket_fd_ != -1)
    {
      close(socket_fd_);
    }
    socket_fd_ = other.socket_fd_;
    other.socket_fd_ = -1;
  }
  return *this;
}

std::size_t UDPSocket::send_to(
  const std::vector<unsigned char>& data,
  const Endpoint& destination)
{
  auto address = create_address(destination);

  return sendto(
    socket_fd_,
    data.data(),
    data.size(),
    0,
    reinterpret_cast<sockaddr*>(&address),
    sizeof(address));
}

std::size_t UDPSocket::send(const std::vector<unsigned char>& data)
{
  if (!destination_.has_value())
  {
    throw std::runtime_error("Destination not set");
  }

  return send_to(data, destination_.value());
}

std::pair<std::size_t, UDPSocket::Endpoint> UDPSocket::receive(
  std::vector<unsigned char>& data)
{
  data.clear();
  sockaddr_in sender_address {};
  socklen_t sender_address_length {sizeof(sender_address)};
  data.resize(DEFAULT_BUFFER_SIZE);
  auto recvlen = recvfrom(
    socket_fd_,
    data.data(),
    data.size(),
    0,
    reinterpret_cast<sockaddr*>(&sender_address),
    &sender_address_length);

  char addr_str[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &sender_address.sin_addr, addr_str, INET_ADDRSTRLEN);
  auto endpoint = Endpoint{addr_str, ntohs(sender_address.sin_port)};

  data.resize(recvlen);

  return std::make_pair(recvlen, endpoint);
}

void UDPSocket::set_receive_timeout(
  const std::chrono::milliseconds& receive_timeout)
{
  struct timeval tv
  {
    .tv_sec = static_cast<time_t>(receive_timeout.count() / 1000),
    .tv_usec = static_cast<suseconds_t>((receive_timeout.count() % 1000) * 1000)};

  if (setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
  {
    throw std::runtime_error("Failed to set receive timeout");
  }

  return;
}

} // namespace IPC