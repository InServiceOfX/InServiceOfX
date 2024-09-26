#ifndef UTILITIES_ARANGE_H
#define UTILITIES_ARANGE_H

#include <algorithm>
#include <cmath> // std::ceil
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Utilities
{

template <typename T>
std::vector<T> arange(const T start, const T stop, const T step=static_cast<T>(1))
{
  std::vector<T> result {};

  if (step == static_cast<T>(0))
  {
    throw std::runtime_error("Cannot divide by 0 with input step.");
  }
  const auto size {std::ceil((stop - start)/step)};

  result.reserve(static_cast<std::size_t>(size));

  // https://en.cppreference.com/w/cpp/algorithm/generate_n
  // OutputIt generate_n(OutputIt first, Size count, Generator g)
  std::generate_n(
    std::back_inserter(result),
    static_cast<std::size_t>(size),
    [n = start, step]() mutable
    {
      auto current = n;
      n += step;
      return current;
    });

  return result;
}

template <typename T>
std::vector<T> arange(const std::size_t N)
{
  return arange<T>(static_cast<T>(0), static_cast<T>(N));
}

} // namespace Utilities

#endif // UTILITIES_ARANGE_H