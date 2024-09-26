#include "DataStructures/Array.h"

#include "gtest/gtest.h"
#include <vector>

using DataStructures::Array;
using std::vector;

namespace GoogleUnitTests
{
namespace DataStructures
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, Constructs)
{
  Array<float> array {4};

  EXPECT_TRUE(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ArrayTests, CopiesToDevice)
{
  const vector<float> host_x {0., 1., 2., 3.};
  Array<float> array {4};
  array.copy_host_input_to_device(host_x);
  vector<float> host_y (4);
  array.copy_device_output_to_host(host_y);
  EXPECT_EQ(host_y.at(0), 0.);
  EXPECT_EQ(host_y.at(1), 1.);
  EXPECT_EQ(host_y.at(2), 2.);
  EXPECT_EQ(host_y.at(3), 3.);
}

} // namespace DataStructures
} // namespace GoogleUnitTests