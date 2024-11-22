#ifndef IPC_UDP_RUNNING_TWO_UDP_TRANSCEIVERS_H
#define IPC_UDP_RUNNING_TWO_UDP_TRANSCEIVERS_H

#include "IPC/UDPSocket.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace IPC
{
namespace UDP
{

class RunningTwoUDPTransceivers
{
  public:

    using Endpoint = IPC::UDPSocket::Endpoint;

    struct Configuration
    {
      Endpoint ip_address_and_port_1;
      Endpoint ip_address_and_port_2;
      Endpoint destination_1;
      Endpoint destination_2;
      std::optional<std::chrono::milliseconds> receive_timeout_1;
      std::optional<std::chrono::milliseconds> receive_timeout_2;

      Configuration(const std::filesystem::path& configuration_file_path);
    };

    RunningTwoUDPTransceivers(const Configuration& configuration);

    virtual ~RunningTwoUDPTransceivers();

    void start_listening();
    void stop_listening();

    void start_listening(const std::chrono::milliseconds timeout_milliseconds);

  protected:

    void listen_continuously_on_transceiver_1();
    void listen_until_timeout_on_transceiver_1(
      const std::chrono::milliseconds timeout_milliseconds);
    void listen_continuously_on_transceiver_2();
    void listen_until_timeout_on_transceiver_2(
      const std::chrono::milliseconds timeout_milliseconds);

  private:

    UDPSocket receiver_1_;
    UDPSocket sender_1_;
    UDPSocket receiver_2_;
    UDPSocket sender_2_;
    std::thread transceiver_thread_1_;
    std::thread transceiver_thread_2_;
    std::atomic<bool> is_transceiver_1_running_;
    std::atomic<bool> is_transceiver_2_running_;
};

} // namespace UDP

} // namespace IPC

#endif // IPC_UDP_RUNNING_UDP_TRANSCEIVER_H
