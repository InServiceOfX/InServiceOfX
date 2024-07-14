#include "DataStructures/Array.h"
#include "Operations/Arithmetic.h"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

using DataStructures::Array;
using Operations::add_scalar;
using std::size_t;
using std::vector;

BOOST_AUTO_TEST_SUITE(Operations)
BOOST_AUTO_TEST_SUITE(Arithmetic_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddScalarAdds)
{
  const size_t example_size {4};

  const vector<float> host_x {0., 1., 2., 3.};
  const size_t threads_per_block {example_size};
  const size_t blocks_per_grid {1};

  Array<float> input {example_size};
  Array<float> output {example_size};
  input.copy_host_input_to_device(host_x);

  add_scalar<float><<<blocks_per_grid, threads_per_block>>>(
    input.elements_,
    output.elements_,
    4.0);

  vector<float> host_y (example_size);

  output.copy_device_output_to_host(host_y);
  BOOST_CHECK_EQUAL(host_y.at(0), 4.);
  BOOST_CHECK_EQUAL(host_y.at(1), 5.);
  BOOST_CHECK_EQUAL(host_y.at(2), 6.);
  BOOST_CHECK_EQUAL(host_y.at(3), 7.);  
}

BOOST_AUTO_TEST_SUITE_END() // Arithmetic_tests
BOOST_AUTO_TEST_SUITE_END() // Operations