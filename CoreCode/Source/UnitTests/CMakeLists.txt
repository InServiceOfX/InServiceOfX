# Dependencies
# Find a boost install with the libraries unit_test_framework.
# Boost filesystem needs boost system.
FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
