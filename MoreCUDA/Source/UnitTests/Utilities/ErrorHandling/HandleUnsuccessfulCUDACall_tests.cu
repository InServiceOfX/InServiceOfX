#include "UnitTests/Utilities/CaptureCerr.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCUDACall.h"

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <string>
#include <string_view>

using UnitTests::Utilities::CaptureCerr;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using std::string;
using std::string_view;

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(ErrorHandling)
BOOST_AUTO_TEST_SUITE(HandleUnsuccessfulCUDACall_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructsWithStringView)
{
  HandleUnsuccessfulCUDACall handler {};
  BOOST_CHECK_EQUAL(
    handler.get_error_message(),
    string_view{"cudaSuccess was not returned."});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithStdString)
{
  const string error_message {"std::string custom error message"};

  HandleUnsuccessfulCUDACall handler {error_message};
  BOOST_CHECK_EQUAL(
    handler.get_error_message(),
    string_view{"std::string custom error message"});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithCString)
{
  const char* error_message {"C string custom error message"};

  HandleUnsuccessfulCUDACall handler {error_message};
  BOOST_CHECK_EQUAL(
    handler.get_error_message(),
    string_view{"C string custom error message"});
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OperatorCallWorks)
{
  CaptureCerr capture_cerr {};

  HandleUnsuccessfulCUDACall handler {"Test CUDA error message"};

  cudaError_t test_error {cudaErrorInvalidValue};

  handler(test_error);

  BOOST_CHECK_EQUAL(handler.get_cuda_error(), test_error);
  BOOST_CHECK_EQUAL(
    capture_cerr.local_oss_.str(),
    "Test CUDA error message (error code invalid argument)!\n");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CheckCUDASuccess)
{
  CaptureCerr capture_cerr {};

  HandleUnsuccessfulCUDACall handler {};

  BOOST_CHECK(handler.is_cuda_success());

  cudaError_t test_error = cudaErrorInvalidValue;
  handler(test_error);

  BOOST_CHECK(!handler.is_cuda_success());
  BOOST_CHECK_EQUAL(
    capture_cerr.local_oss_.str(),
    "cudaSuccess was not returned. (error code invalid argument)!\n");
}

BOOST_AUTO_TEST_SUITE_END() // HandleUnsuccessfulCUDACall_tests
BOOST_AUTO_TEST_SUITE_END() // ErrorHandling
BOOST_AUTO_TEST_SUITE_END() // Utilities