#include "RunningUDPTransceiver.h"
#include "IPC/UDPSocket.h"

#include <chrono>
#include <optional>
#include <thread>
#include <vector>

using IPC::UDPSocket;
using Endpoint = UDPSocket::Endpoint;
using UDPSocketConfiguration = UDPSocket::Configuration;
using std::nullopt;
using std::thread;
using std::vector;

namespace IPC
{

namespace UDP
{

RunningUDPTransceiver::RunningUDPTransceiver(
  const UDPSocketConfiguration& configuration
):
  receiver_{
    UDPSocketConfiguration{
      configuration.ip_address_and_port_,
      configuration.receive_timeout_,
      nullopt}},
  sender_{UDPSocketConfiguration{nullopt, nullopt, configuration.destination_}},
  is_transceiver_running_{false}
{}


RunningUDPTransceiver::~RunningUDPTransceiver()
{
  stop_listening();
}

void RunningUDPTransceiver::start_listening()
{
  is_transceiver_running_ = true;
  transceiver_thread_ = thread{&RunningUDPTransceiver::listen_continuously, this};
}

void RunningUDPTransceiver::stop_listening()
{
  is_transceiver_running_ = false;
  if (transceiver_thread_.joinable())
  {
    transceiver_thread_.join();
  }
}

void RunningUDPTransceiver::start_listening(
  const std::chrono::milliseconds timeout_milliseconds)
{
  is_transceiver_running_ = true;
  transceiver_thread_ = thread{
    &RunningUDPTransceiver::listen_until_timeout,
    this,
    timeout_milliseconds};
}

void RunningUDPTransceiver::listen_continuously()
{
  while (is_transceiver_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_.receive(data);

    if (result.first > 0)
    {
      sender_.send(data);
    }
  }
}

void RunningUDPTransceiver::listen_until_timeout(
  const std::chrono::milliseconds timeout_milliseconds)
{
  auto start_time = std::chrono::steady_clock::now();
  while (is_transceiver_running_)
  {
    vector<unsigned char> data {};
    auto result = receiver_.receive(data);

    if (result.first > 0)
    {
      sender_.send(data);
    }

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = current_time - start_time;
    if (elapsed_time >= timeout_milliseconds)
    {
      is_transceiver_running_ = false;
      break;
    }
  }
}

} // namespace UDP

} // namespace IPC