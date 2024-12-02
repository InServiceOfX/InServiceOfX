//------------------------------------------------------------------------------
/// \brief Help capture std::cerr standard output.
///-----------------------------------------------------------------------------
#ifndef UTILITIES_CAPTURE_CERR_H
#define UTILITIES_CAPTURE_CERR_H

#include <sstream> // std::ostringstream
#include <utility> // std::pair

namespace Utilities
{

//------------------------------------------------------------------------------
/// \return The original stream buffer that we've displaced, so that it can be
/// used again to restore the std::cout buffer.
//------------------------------------------------------------------------------
std::streambuf* capture_cerr(std::ostringstream& local_oss);

class CaptureCerr
{
  public:

    // Buffer to capture cout; it essentially displaces the stream of std::cout.
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

#endif // UTILITIES_CAPTURE_CERR_H