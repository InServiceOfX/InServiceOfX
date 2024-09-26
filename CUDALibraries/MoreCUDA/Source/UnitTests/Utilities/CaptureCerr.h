#ifndef UNIT_TESTS_UTILITIES_CAPTURE_CERR_H
#define UNIT_TESTS_UTILITIES_CAPTURE_CERR_H

#include <sstream> // std::ostringstream

namespace UnitTests
{
namespace Utilities
{

//------------------------------------------------------------------------------
/// \return The original stream buffer that we've displaced, so that it can be
/// used again to restore the std::cout buffer.
//------------------------------------------------------------------------------
std::streambuf* capture_cerr(std::ostringstream& local_oss);

//------------------------------------------------------------------------------
/// \brief Help capture std::cerr standard output.
///-----------------------------------------------------------------------------
class CaptureCerr
{
  public:

    // Buffer to capture cerr; it essentially displaces the stream of std::cerr.
    std::ostringstream local_oss_;

    CaptureCerr();

    ~CaptureCerr();

  protected:

    void restore_cerr();

    void capture_locally();

  private:

    // Save previous buffer.  
    std::streambuf* cerr_buffer_ptr_;
};

} // namespace Utilities
} // namespace UnitTests

#endif // UNIT_TESTS_UTILITIES_CAPTURE_CERR_H