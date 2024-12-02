#include "Utilities/CaptureCerr.h"

#include <iostream>
#include <sstream> // std::ostringstream
#include <streambuf>

using std::cerr;
using std::ostringstream;
using std::streambuf;

namespace Utilities
{

//------------------------------------------------------------------------------
/// \details
/// cf. https://en.cppreference.com/w/cpp/io/basic_ostringstream
/// ostringstream effectively stores an instance of std string and performs
/// output operations to it.
/// rdbuf() returns associated stream buffer.
/// rdbuf(streambuf* sb) sets associated stream buffer to sb. Returns associated
/// stream buffer before operation. If there's no associated stream buffer,
/// returns null pointer.
//------------------------------------------------------------------------------

streambuf* capture_cerr(ostringstream& local_oss)
{
  streambuf* cerr_buffer {cerr.rdbuf()};

  cerr.rdbuf(local_oss.rdbuf());

  return cerr_buffer;
}

CaptureCerr::CaptureCerr():
  local_oss_{},
  cerr_buffer_ptr_{cerr.rdbuf()} // Save previous buffer.
{
  capture_locally();
}

CaptureCerr::~CaptureCerr()
{
  this->restore_cerr();
}

void CaptureCerr::capture_locally()
{
  cerr.rdbuf(local_oss_.rdbuf());
}

void CaptureCerr::restore_cerr()
{
  cerr.rdbuf(cerr_buffer_ptr_);
}

} // namespace Utilities