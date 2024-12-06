#ifndef STREAM_MANAGEMENT_STREAM_H
#define STREAM_MANAGEMENT_STREAM_H

namespace StreamManagement
{

class Stream
{
  public:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da
    /// 
    //--------------------------------------------------------------------------
    Stream();

    ~Stream();

    cudaStream_t stream_;

  protected:

    bool create_stream();
    bool destroy_stream();
};

} // namespace StreamManagement

#endif // STREAM_MANAGEMENT_STREAM_H
