#include "DataStructures/Array.h"

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <vector>

using DataStructures::Array;
using std::vector;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Array_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  Array<float> array {4};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CopiesToDevice)
{
  const vector<float> host_x {0., 1., 2., 3.};
  Array<float> array {4};
  array.copy_host_input_to_device(host_x);
  vector<float> host_y (4);
  array.copy_device_output_to_host(host_y);
  BOOST_CHECK_EQUAL(host_y.at(0), 0.);
  BOOST_CHECK_EQUAL(host_y.at(1), 1.);
  BOOST_CHECK_EQUAL(host_y.at(2), 2.);
  BOOST_CHECK_EQUAL(host_y.at(3), 3.);  
}

BOOST_AUTO_TEST_SUITE_END() // Array_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures