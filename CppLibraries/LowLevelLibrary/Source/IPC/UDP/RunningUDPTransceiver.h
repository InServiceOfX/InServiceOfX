#ifndef IPC_UDP_RUNNING_UDP_TRANSCEIVER_H
#define IPC_UDP_RUNNING_UDP_TRANSCEIVER_H

#include "IPC/UDPSocket.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace IPC
{
namespace UDP
{

class RunningUDPTransceiver
{
  public:

    RunningUDPTransceiver(const UDPSocket::Configuration& configuration);

    virtual ~RunningUDPTransceiver();

    void start_listening();
    void stop_listening();

    void start_listening(const std::chrono::milliseconds timeout_milliseconds);

  protected:

    void listen_continuously();
    void listen_until_timeout(
      const std::chrono::milliseconds timeout_milliseconds);

  private:

    UDPSocket receiver_;
    UDPSocket sender_;
    std::thread transceiver_thread_;
    std::atomic<bool> is_transceiver_running_;
};

} // namespace UDP

} // namespace IPC

#endif // IPC_UDP_RUNNING_UDP_TRANSCEIVER_H
