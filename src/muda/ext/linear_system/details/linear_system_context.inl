#include <muda/check/check_cublas.h>
#include <muda/check/check_cusolver.h>
#include <muda/check/check_cusparse.h>
namespace muda
{
MUDA_INLINE LinearSystemContext::LinearSystemContext(const LinearSystemContextCreateInfo& info)
    : m_stream(info.stream)
    , m_create_info(info)
{
    checkCudaErrors(cusparseCreate(&m_cusparse));
    checkCudaErrors(cublasCreate(&m_cublas));
    checkCudaErrors(cusolverDnCreate(&m_cusolver_dn));
    checkCudaErrors(cusparseSetStream(m_cusparse, m_stream));
    checkCudaErrors(cublasSetStream(m_cublas, m_stream));
    checkCudaErrors(cusolverDnSetStream(m_cusolver_dn, m_stream));
    checkCudaErrors(cusolverSpCreate(&m_cusolver_sp));
    checkCudaErrors(cusolverSpSetStream(m_cusolver_sp, m_stream));

    checkCudaErrors(cusparseSetPointerMode(m_cusparse, CUSPARSE_POINTER_MODE_HOST));
    checkCudaErrors(cublasSetPointerMode(m_cublas, CUBLAS_POINTER_MODE_HOST));
    m_pointer_mode_device = false;

    m_buffers.emplace_back(info.buffer_byte_size_base);
}

MUDA_INLINE LinearSystemContext::~LinearSystemContext()
{
    if(m_cusparse)
        checkCudaErrors(cusparseDestroy(m_cusparse));
    if(m_cublas)
        checkCudaErrors(cublasDestroy(m_cublas));
    if(m_cusolver_dn)
        checkCudaErrors(cusolverDnDestroy(m_cusolver_dn));
    if(m_cusolver_sp)
        checkCudaErrors(cusolverSpDestroy(m_cusolver_sp));
}

MUDA_INLINE void LinearSystemContext::stream(cudaStream_t stream)
{
    m_stream = stream;
    checkCudaErrors(cusparseSetStream(m_cusparse, m_stream));
    checkCudaErrors(cublasSetStream(m_cublas, m_stream));
    checkCudaErrors(cusolverDnSetStream(m_cusolver_dn, m_stream));
    checkCudaErrors(cusolverSpSetStream(m_cusolver_sp, m_stream));
}

MUDA_INLINE void LinearSystemContext::shrink_temp_buffers()
{
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    // get the largest buffer
    auto first = m_buffers.begin();
    auto last  = std::prev(m_buffers.end());
    std::iter_swap(first, last);
    // remove all but the largest buffer
    m_buffers.resize(1);
}

MUDA_INLINE void LinearSystemContext::set_pointer_mode_device()
{
    if(m_pointer_mode_device)
        return;
    checkCudaErrors(cusparseSetPointerMode(m_cusparse, CUSPARSE_POINTER_MODE_DEVICE));
    checkCudaErrors(cublasSetPointerMode(m_cublas, CUBLAS_POINTER_MODE_DEVICE));
}
MUDA_INLINE void LinearSystemContext::set_pointer_mode_host()
{
    if(!m_pointer_mode_device)
        return;
    checkCudaErrors(cusparseSetPointerMode(m_cusparse, CUSPARSE_POINTER_MODE_HOST));
    checkCudaErrors(cublasSetPointerMode(m_cublas, CUBLAS_POINTER_MODE_HOST));
}
MUDA_INLINE void LinearSystemContext::add_sync_callback(std::function<void()>&& callback)
{
    m_sync_callbacks.emplace_back(std::move(callback));
}

MUDA_INLINE span<std::byte> LinearSystemContext::temp_host_buffer(size_t size)
{
    for(auto& b : m_host_buffers)
        if(b.size() >= size)
            return span<std::byte>{b.data(), size};
    auto base = m_create_info.buffer_byte_size_base;
    // round up to multiple of base
    auto rounded_size = ((size + base - 1) / base) * base;
    return span<std::byte>{m_host_buffers.emplace_back(rounded_size).data(), size};
}

MUDA_INLINE BufferView<std::byte> LinearSystemContext::temp_buffer(size_t size)
{
    for(auto& b : m_buffers)
        if(b.size() >= size)
            return b.view(0, size);
    auto base = m_create_info.buffer_byte_size_base;
    // round up to multiple of base
    auto rounded_size = ((size + base - 1) / base) * base;
    return m_buffers.emplace_back(rounded_size).view(0, size);
}

template <typename T>
BufferView<T> LinearSystemContext::temp_buffer(size_t size)
{
    BufferView<std::byte> bbuf = temp_buffer(size * sizeof(T));
    return BufferView<T>{reinterpret_cast<T*>(bbuf.data()), 0, size};
}

template <typename T>
span<T> LinearSystemContext::temp_host_buffer(size_t size)
{
    span<std::byte> bbuf = temp_host_buffer(size * sizeof(T));
    return span<T>{reinterpret_cast<T*>(bbuf.data()), size};
}

template <typename T>
std::vector<T*> LinearSystemContext::temp_buffers(size_t size_in_buffer, size_t num_buffer)
{
    BufferView<T>   total = temp_buffer<T>(size_in_buffer * num_buffer);
    std::vector<T*> ret(num_buffer);
    for(int i = 0; i < num_buffer; ++i)
        ret[i] = total.data(i * size_in_buffer);
    return ret;
}

template <typename T>
std::vector<T*> LinearSystemContext::temp_host_buffers(size_t size_in_buffer, size_t num_buffer)
{
    span<T>         total = temp_host_buffer<T>(size_in_buffer * num_buffer);
    std::vector<T*> ret(num_buffer);
    for(int i = 0; i < num_buffer; ++i)
        ret[i] = total.data() + i * size_in_buffer;
    return ret;
}

MUDA_INLINE void LinearSystemContext::sync()
{
    on(m_stream).wait();
    // wait and reduce temp buffers
    if(m_buffers.size() > 1)
        shrink_temp_buffers();
    // call callbacks
    for(auto& cb : m_sync_callbacks)
        cb();
    m_sync_callbacks.clear();
}

}  // namespace muda