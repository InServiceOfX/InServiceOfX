#include "RunningTwoUDPTransceivers.h"
#include "IPC/UDPSocket.h"
#include "Utilities/Parsers/parse_simple_map.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

using UDPSocketConfiguration = IPC::UDPSocket::Configuration;
using Utilities::Parsers::parse_simple_map;
using std::nullopt;
using std::thread;
using std::vector;

namespace IPC
{
namespace UDP
{

RunningTwoUDPTransceivers::Configuration::Configuration(
  const std::filesystem::path& configuration_file_path
):
  ip_address_and_port_1{},
  ip_address_and_port_2{},
  destination_1{},
  destination_2{},
  receive_timeout_1{},
  receive_timeout_2{}
{
  const auto configuration_map = parse_simple_map(configuration_file_path);

  auto iterator_ip_address_1 = configuration_map.find("ip_address_1");
  auto iterator_port_1 = configuration_map.find("port_1");

  if (iterator_ip_address_1 == configuration_map.end() ||
      iterator_port_1 == configuration_map.end())
  {
    throw std::runtime_error("ip_address_1 and port_1 must be specified");
  }
  else
  {
    ip_address_and_port_1 = Endpoint{
      iterator_ip_address_1->second,
      static_cast<uint16_t>(std::stoul(iterator_port_1->second))
    };
  }

  auto iterator_ip_address_2 = configuration_map.find("ip_address_2");
  auto iterator_port_2 = configuration_map.find("port_2");

  if (iterator_ip_address_2 == configuration_map.end() ||
      iterator_port_2 == configuration_map.end())
  {
    throw std::runtime_error("ip_address_2 and port_2 must be specified");
  }
  else
  {
    ip_address_and_port_2 = Endpoint{
      iterator_ip_address_2->second,
      static_cast<uint16_t>(std::stoul(iterator_port_2->second))
    };
  }

  auto iterator_receive_timeout_1 = configuration_map.find("receive_timeout_1");
  if (iterator_receive_timeout_1 == configuration_map.end())
  {
    receive_timeout_1 = std::nullopt;
  }
  else
  {
    receive_timeout_1 = std::chrono::milliseconds(
      std::stoul(iterator_receive_timeout_1->second));
  }

  auto iterator_receive_timeout_2 = configuration_map.find("receive_timeout_2");
  if (iterator_receive_timeout_2 == configuration_map.end())
  {
    receive_timeout_2 = std::nullopt;
  }
  else
  {
    receive_timeout_2 = std::chrono::milliseconds(
      std::stoul(iterator_receive_timeout_2->second));
  }

  auto iterator_destination_ip_address_1 = configuration_map.find(
    "destination_ip_address_1");
  auto iterator_destination_port_1 = configuration_map.find(
    "destination_port_1");

  if (iterator_destination_ip_address_1 == configuration_map.end() ||
      iterator_destination_port_1 == configuration_map.end())
  {
    throw std::runtime_error(
      "destination_ip_address_1 and destination_port_1 must be specified");
  }
  else
  {
    destination_1 = Endpoint{
      iterator_destination_ip_address_1->second,
      static_cast<uint16_t>(std::stoul(iterator_destination_port_1->second))
    };
  }

  auto iterator_destination_ip_address_2 = configuration_map.find(
    "destination_ip_address_2");
  auto iterator_destination_port_2 = configuration_map.find(
    "destination_port_2");

  if (iterator_destination_ip_address_2 == configuration_map.end() ||
      iterator_destination_port_2 == configuration_map.end())
  {
    throw std::runtime_error(
      "destination_ip_address_2 and destination_port_2 must be specified");
  }
  else
  {
    destination_2 = Endpoint{
      iterator_destination_ip_address_2->second,
      static_cast<uint16_t>(std::stoul(iterator_destination_port_2->second))
    };
  }
}

RunningTwoUDPTransceivers::RunningTwoUDPTransceivers(
  const Configuration& configuration
):
  receiver_1_{
    UDPSocketConfiguration{
      configuration.ip_address_and_port_1,
      configuration.receive_timeout_1,
      nullopt}},
  sender_1_{
    UDPSocketConfiguration{
      nullopt,
      nullopt,
      configuration.destination_1}},
  receiver_2_{
    UDPSocketConfiguration{
      configuration.ip_address_and_port_2,
      configuration.receive_timeout_2,
      nullopt}},
  sender_2_{
    UDPSocketConfiguration{
      nullopt,
      nullopt,
      configuration.destination_2}},
  is_transceiver_1_running_{false},
  is_transceiver_2_running_{false}
{}


RunningTwoUDPTransceivers::~RunningTwoUDPTransceivers()
{
  stop_listening();
}

void RunningTwoUDPTransceivers::start_listening()
{
  is_transceiver_1_running_ = true;
  is_transceiver_2_running_ = true;

  transceiver_thread_1_ = thread{
    &RunningTwoUDPTransceivers::listen_continuously_on_transceiver_1, this};
  transceiver_thread_2_ = thread{
    &RunningTwoUDPTransceivers::listen_continuously_on_transceiver_2, this};
}

void RunningTwoUDPTransceivers::start_listening(
  const std::chrono::milliseconds timeout_milliseconds)
{
  is_transceiver_1_running_ = true;
  is_transceiver_2_running_ = true;

  transceiver_thread_1_ = thread{
    &RunningTwoUDPTransceivers::listen_until_timeout_on_transceiver_1,
    this,
    timeout_milliseconds};
  transceiver_thread_2_ = thread{
    &RunningTwoUDPTransceivers::listen_until_timeout_on_transceiver_2,
    this,
    timeout_milliseconds};
}

void RunningTwoUDPTransceivers::stop_listening()
{
  is_transceiver_1_running_ = false;
  is_transceiver_2_running_ = false;
  if (transceiver_thread_1_.joinable())
  {
    transceiver_thread_1_.join();
  }
  if (transceiver_thread_2_.joinable())
  {
    transceiver_thread_2_.join();
  }
}

void RunningTwoUDPTransceivers::listen_continuously_on_transceiver_1()
{
  while (is_transceiver_1_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_1_.receive(data);

    if (result.first > 0)
    {
      sender_1_.send(data);
    }
  }
}


void RunningTwoUDPTransceivers::listen_continuously_on_transceiver_2()
{
  while (is_transceiver_2_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_2_.receive(data);

    if (result.first > 0)
    {
      sender_2_.send(data);
    }
  }
}

void RunningTwoUDPTransceivers::listen_until_timeout_on_transceiver_1(
  const std::chrono::milliseconds timeout_milliseconds)
{
  auto start_time = std::chrono::steady_clock::now();
  while (is_transceiver_1_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_1_.receive(data);

    if (result.first > 0)
    {
      sender_1_.send(data);
    }

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = current_time - start_time;
    if (elapsed_time >= timeout_milliseconds)
    {
      is_transceiver_1_running_ = false;
      break;
    }
  }
}

void RunningTwoUDPTransceivers::listen_until_timeout_on_transceiver_2(
  const std::chrono::milliseconds timeout_milliseconds)
{
  auto start_time = std::chrono::steady_clock::now();
  while (is_transceiver_2_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_2_.receive(data);

    if (result.first > 0)
    {
      sender_2_.send(data);
    }

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = current_time - start_time;
    if (elapsed_time >= timeout_milliseconds)
    {
      is_transceiver_2_running_ = false;
      break;
    }
  }
}

} // namespace UDP

} // namespace IPC