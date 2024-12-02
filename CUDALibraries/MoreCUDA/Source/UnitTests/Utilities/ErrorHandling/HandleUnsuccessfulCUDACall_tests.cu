#include "Utilities/CaptureCerr.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <string>
#include <string_view>

using Utilities::CaptureCerr;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using std::string;
using std::string_view;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleUnsuccessfulCUDACallTests, DefaultConstructsWithStringView)
{
  HandleUnsuccessfulCUDACall handler {};
  EXPECT_EQ(
    handler.get_error_message(),
    string_view{"cudaSuccess was not returned."});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleUnsuccessfulCUDACallTests, ConstructsWithStdString)
{
  const string error_message {"std::string custom error message"};

  HandleUnsuccessfulCUDACall handler {error_message};
  EXPECT_EQ(
    handler.get_error_message(),
    string_view{"std::string custom error message"});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleUnsuccessfulCUDACallTests, ConstructsWithCString)
{
  const char* error_message {"C string custom error message"};

  HandleUnsuccessfulCUDACall handler {error_message};
  EXPECT_EQ(
    handler.get_error_message(),
    string_view{"C string custom error message"});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleUnsuccessfulCUDACallTests, OperatorCallWorks)
{
  CaptureCerr capture_cerr {};

  HandleUnsuccessfulCUDACall handler {"Test CUDA error message"};

  cudaError_t test_error {cudaErrorInvalidValue};

  handler(test_error);

  EXPECT_EQ(handler.get_cuda_error(), test_error);
  EXPECT_EQ(
    capture_cerr.local_oss_.str(),
    "Test CUDA error message (error code invalid argument)!\n");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleUnsuccessfulCUDACallTests, CheckCUDASuccess)
{
  CaptureCerr capture_cerr {};

  HandleUnsuccessfulCUDACall handler {};

  EXPECT_TRUE(handler.is_cuda_success());

  cudaError_t test_error = cudaErrorInvalidValue;
  handler(test_error);

  EXPECT_TRUE(!handler.is_cuda_success());
  EXPECT_EQ(
    capture_cerr.local_oss_.str(),
    "cudaSuccess was not returned. (error code invalid argument)!\n");
}

} // namespace ErrorHandling
} // namespace Utilities
} // namespace GoogleUnitTests