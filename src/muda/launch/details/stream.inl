namespace muda
{
MUDA_INLINE Stream::Stream(Stream::Flag f)
{
    checkCudaErrors(cudaStreamCreateWithFlags(&m_handle, static_cast<unsigned int>(f)));
}

MUDA_INLINE void Stream::wait() const
{
    checkCudaErrors(cudaStreamSynchronize(m_handle));
}

MUDA_INLINE void Stream::begin_capture(cudaStreamCaptureMode mode) const
{
    checkCudaErrors(cudaStreamBeginCapture(m_handle, mode));
}

MUDA_INLINE void Stream::end_capture(cudaGraph_t* graph) const
{
    checkCudaErrors(cudaStreamEndCapture(m_handle, graph));
}

MUDA_INLINE MUDA_DEVICE Stream::TailLaunch::operator cudaStream_t() const
{
#ifdef __CUDA_ARCH__
    return cudaStreamTailLaunch;
#endif
}

MUDA_INLINE MUDA_DEVICE Stream::FireAndForget::operator cudaStream_t() const
{
#ifdef __CUDA_ARCH__
    return cudaStreamFireAndForget;
#endif
}

MUDA_INLINE MUDA_DEVICE Stream::GraphTailLaunch::operator cudaStream_t() const
{
#ifdef __CUDA_ARCH__
    return cudaStreamGraphTailLaunch;
#endif
}

MUDA_INLINE MUDA_DEVICE Stream::GraphFireAndForget::operator cudaStream_t() const
{
#ifdef __CUDA_ARCH__
    return cudaStreamGraphFireAndForget;
#endif
}

MUDA_INLINE Stream::~Stream()
{
    if(m_handle)
        checkCudaErrors(cudaStreamDestroy(m_handle));
}

MUDA_INLINE Stream::Stream(Stream&& o) MUDA_NOEXCEPT : m_handle(o.m_handle)
{
    o.m_handle = nullptr;
}

MUDA_INLINE Stream& Stream::operator=(Stream&& o) MUDA_NOEXCEPT
{
    if(this == &o)
        return *this;

    if(m_handle)
        checkCudaErrors(cudaStreamDestroy(m_handle));

    m_handle   = o.m_handle;
    o.m_handle = nullptr;
    return *this;
}
}  // namespace muda