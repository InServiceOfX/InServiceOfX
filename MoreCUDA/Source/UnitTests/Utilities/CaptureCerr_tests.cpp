#include "UnitTests/Utilities/CaptureCerr.h"

#include <boost/test/unit_test.hpp>
#include <iostream> // std::cerr

using std::cerr;
using UnitTests::Utilities::CaptureCerr;

BOOST_AUTO_TEST_SUITE(UnitTests)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(CaptureCerr_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	CaptureCerr capture_cerr {};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CaptureLocallyUponConstruction)
{
  CaptureCerr capture_cerr {};
  cerr << "\n Testing Testing \n";

  BOOST_CHECK_EQUAL(capture_cerr.local_oss_.str(), "\n Testing Testing \n");
}

BOOST_AUTO_TEST_SUITE_END() // CaptureCerr_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // UnitTests