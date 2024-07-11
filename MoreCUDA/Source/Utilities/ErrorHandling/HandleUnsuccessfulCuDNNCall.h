#ifndef UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H
#define UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H

#include <cudnn.h>
#include <string>
#include <string_view>

namespace Utilities
{
namespace ErrorHandling
{

class HandleUnsuccessfulCuDNNCall
{
  public:

    inline static const std::string_view default_error_message_ {
      "cuDNN status Success was not returned."};

    HandleUnsuccessfulCuDNNCall(
      const std::string_view error_message = default_error_message_);

    HandleUnsuccessfulCuDNNCall(const std::string& error_message);

    HandleUnsuccessfulCuDNNCall(const char* error_message);

    ~HandleUnsuccessfulCuDNNCall() = default;

    inline bool is_success() const
    {
      return status_ == CUDNN_STATUS_SUCCESS;
    }

    void operator()(const cudnnStatus_t cuDNN_status);

    void operator()(
      const cudnnStatus_t cuDNN_status,
      const char* file,
      const int line);

    inline cudnnStatus_t get_status() const
    {
      return status_;
    }

    inline std::string_view get_error_message() const
    {
      return error_message_;
    }

  private:

    std::string_view error_message_;

    cudnnStatus_t status_;
};

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_HANDLE_UNSUCCESSFUL_CUDNN_CALL_H
