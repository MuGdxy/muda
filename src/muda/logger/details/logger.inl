#include <device_atomic_functions.h>
#include <algorithm>
#include <sstream>
#include <muda/mstl/span.h>
#include <muda/debug.h>
namespace muda
{
MUDA_INLINE MUDA_DEVICE LoggerViewer::Proxy::Proxy(LoggerViewer& viewer)
    : m_viewer(viewer)
{
    MUDA_KERNEL_ASSERT(m_viewer.m_buffer_view_data && m_viewer.m_meta_data_view_data,
                       "LoggerViewer is not initialized");
    m_log_id = atomicAdd(&(m_viewer.m_offset_view->log_id), 1u);
}

MUDA_INLINE MUDA_DEVICE LoggerViewer::Proxy& LoggerViewer::Proxy::operator<<(const char* str)
{
    auto strlen = [](const char* s)
    {
        size_t len = 0;
        while(*s++)
            ++len;
        return len;
    };
    auto size = strlen(str) + 1;

    details::LoggerMetaData meta;
    meta.type = details::LoggerBasicType::String;
    meta.size = size;
    meta.id   = m_log_id;
    m_viewer.push_data(meta, str);
    return *this;
}

template <typename T>
MUDA_INLINE MUDA_DEVICE LoggerViewer::Proxy LoggerViewer::operator<<(const T& t)
{
    Proxy p(*this);
    p << t;
    return p;
}

MUDA_INLINE MUDA_DEVICE uint32_t next_idx(uint32_t* data_offset, uint32_t size, uint32_t total_size)
{

    uint32_t old = *data_offset;
    if(old + size >= total_size)
        return ~0u;
    uint32_t assumed;
    do
    {
        assumed         = old;
        auto new_offset = old + size;
        old             = atomicCAS(data_offset, assumed, new_offset);
        if(old + size >= total_size)
        {
            old = ~0u;
            break;
        }
    } while(assumed != old);
    return old;
}

MUDA_INLINE MUDA_DEVICE LoggerViewer::Proxy LoggerViewer::operator<<(const char* s)
{
    Proxy p(*this);
    std::move(p) << s;
    return std::move(p);
}

MUDA_INLINE MUDA_DEVICE uint32_t LoggerViewer::next_meta_data_idx() const
{
    auto idx = next_idx(&(m_offset_view->meta_data_offset), 1u, m_meta_data_view_size);
    if(idx == ~0u)
    {
        atomicCAS(&(m_offset_view->exceed_meta_data), 0u, 1u);
        return ~0u;
    }
    return idx;
}

MUDA_INLINE MUDA_DEVICE uint32_t LoggerViewer::next_buffer_idx(uint32_t size) const
{
    auto idx = next_idx(&(m_offset_view->buffer_offset), size, m_buffer_view_size);
    if(idx == ~0u)
    {
        atomicCAS(&(m_offset_view->exceed_buffer), 0u, 1u);
        return ~0u;
    }
    return idx;
}

MUDA_INLINE MUDA_DEVICE bool LoggerViewer::push_data(details::LoggerMetaData meta,
                                                     const void* data)
{
    auto meta_idx = next_meta_data_idx();
    if(meta_idx == ~0u)
        return false;
    auto buffer_idx = next_buffer_idx(meta.size);
    if(buffer_idx == ~0u)
    {
        meta.exceeded                   = true;
        m_meta_data_view_data[meta_idx] = meta;
        return false;
    }
    meta.offset                     = buffer_idx;
    m_meta_data_view_data[meta_idx] = meta;
    for(int i = 0; i < meta.size; ++i)
        m_buffer_view_data[buffer_idx + i] = reinterpret_cast<const char*>(data)[i];
    return true;
}

MUDA_INLINE void details::LoggerMetaData::put(std::ostream& os, char* buffer) const
{
#define MUDA_PUT_CASE(EnumT, T)                                                \
    case LoggerBasicType::EnumT:                                               \
        os << *reinterpret_cast<const T*>(buffer + offset);                    \
        break;

    switch(type)
    {
        case LoggerBasicType::String:
            os << buffer + offset;
            break;
            MUDA_PUT_CASE(Int8, int8_t);
            MUDA_PUT_CASE(Int16, int16_t);
            MUDA_PUT_CASE(Int32, int32_t);
            MUDA_PUT_CASE(Int64, int64_t);
            MUDA_PUT_CASE(UInt8, uint8_t);
            MUDA_PUT_CASE(UInt16, uint16_t);
            MUDA_PUT_CASE(UInt32, uint32_t);
            MUDA_PUT_CASE(UInt64, uint64_t);
            MUDA_PUT_CASE(Float, float);
            MUDA_PUT_CASE(Double, double);
        default:
            MUDA_ERROR_WITH_LOCATION("Unknown type");
            break;
    }
#undef MUDA_PUT_CASE
}


MUDA_INLINE Logger::Logger(LoggerViewer* global_viewer, size_t meta_size, size_t buffer_size)
    : m_meta_data(nullptr)
    , m_meta_data_size(meta_size)
    , m_h_meta_data(meta_size)
    , m_buffer(nullptr)
    , m_buffer_size(buffer_size)
    , m_h_buffer(buffer_size)
    , m_offset(nullptr)
    , m_h_offset()
    , m_log_viewer_ptr(global_viewer)
{
    checkCudaErrors(cudaMalloc(&m_meta_data, sizeof(details::LoggerMetaData) * meta_size));
    checkCudaErrors(cudaMalloc(&m_buffer, sizeof(char) * buffer_size));
    checkCudaErrors(cudaMalloc(&m_offset, sizeof(details::LoggerOffset)));
    upload();
}

MUDA_INLINE void Logger::retrieve(std::ostream& os)
{
    download();
    auto meta_data_span =
        span<details::LoggerMetaData>{m_h_meta_data}.subspan(0, m_h_offset.meta_data_offset);
    std::stable_sort(meta_data_span.begin(),
                     meta_data_span.end(),
                     [](const details::LoggerMetaData& a, const details::LoggerMetaData& b)
                     { return a.id < b.id; });

    std::stringstream ss;
    for(const auto& meta_data : meta_data_span)
    {
        if(meta_data.exceeded)
            ss << "[log_id " << meta_data.id << ": buffer exceeded]";
        else
            meta_data.put(ss, m_h_buffer.data());
    }
    expand_if_needed();
    os << ss.str();
    upload();
}

MUDA_INLINE void Logger::expand_meta_data()
{
    auto                     new_size = m_meta_data_size * 2;
    details::LoggerMetaData* new_meta_data;
    checkCudaErrors(cudaMalloc(&new_meta_data, new_size * sizeof(details::LoggerMetaData)));
    checkCudaErrors(cudaFree(m_meta_data));
    m_meta_data      = new_meta_data;
    m_meta_data_size = new_size;
}

MUDA_INLINE void Logger::expand_buffer()
{
    auto  new_size = m_buffer_size * 2;
    char* new_buffer;
    checkCudaErrors(cudaMalloc(&new_buffer, new_size * sizeof(char)));
    checkCudaErrors(cudaFree(m_buffer));
    m_buffer      = new_buffer;
    m_buffer_size = new_size;
}

MUDA_INLINE void Logger::upload()
{
    // reset
    m_h_offset = {};
    checkCudaErrors(cudaMemcpyAsync(m_offset, &m_h_offset, sizeof(m_h_offset), cudaMemcpyHostToDevice));

    m_viewer.m_buffer_view_data    = m_buffer;
    m_viewer.m_buffer_view_size    = m_buffer_size;
    m_viewer.m_meta_data_view_data = m_meta_data;
    m_viewer.m_meta_data_view_size = m_meta_data_size;
    m_viewer.m_offset_view         = m_offset;
    if(m_log_viewer_ptr)
    {
        checkCudaErrors(cudaMemcpyAsync(
            m_log_viewer_ptr, &m_viewer, sizeof(m_viewer), cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaStreamSynchronize(nullptr));
}

MUDA_INLINE void Logger::download()
{
    // copy back
    checkCudaErrors(cudaMemcpy(&m_h_offset, m_offset, sizeof(m_h_offset), cudaMemcpyDeviceToHost));

    if(m_h_offset.buffer_offset > 0)
    {
        checkCudaErrors(cudaMemcpyAsync(m_h_buffer.data(),
                                        m_buffer,
                                        m_h_offset.buffer_offset * sizeof(char),
                                        cudaMemcpyDeviceToHost));
    }

    if(m_h_offset.meta_data_offset > 0)
    {
        checkCudaErrors(cudaMemcpyAsync(m_h_meta_data.data(),
                                        m_meta_data,
                                        m_h_offset.meta_data_offset * sizeof(details::LoggerMetaData),
                                        cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaStreamSynchronize(nullptr));
}

MUDA_INLINE void Logger::expand_if_needed()
{
    if(m_h_offset.exceed_meta_data)
    {
        auto old_size = m_meta_data_size;
        expand_meta_data();
        auto new_size = m_meta_data_size;

        m_h_offset.exceed_meta_data = 0;
        MUDA_KERNEL_WARN_WITH_LOCATION(
            "Logger meta data buffer expanded %d => %d", old_size, new_size);
    }
    if(m_h_offset.exceed_buffer)
    {
        auto old_size = m_buffer_size;
        expand_buffer();
        auto new_size = m_buffer_size;

        m_h_offset.exceed_buffer = 0;
        MUDA_KERNEL_WARN_WITH_LOCATION("Logger buffer expanded %d => %d", old_size, new_size);
    }
}

MUDA_INLINE Logger::~Logger()
{
    checkCudaErrors(cudaFree(m_buffer));
    checkCudaErrors(cudaFree(m_meta_data));
    checkCudaErrors(cudaFree(m_offset));
}

//MUDA_INLINE Logger& Logger::instance()
//{
//    static std::unique_ptr<Logger> logger = nullptr;
//    if(!logger)
//    {
//        LoggerViewer* log_viewer_ptr = nullptr;
//        checkCudaErrors(cudaGetSymbolAddress((void**)&log_viewer_ptr, muda::cout));
//        logger = std::make_unique<Logger>(log_viewer_ptr);
//    }
//    return *logger;
//}

//MUDA_INLINE std::mutex& Logger::mutex()
//{
//    static std::mutex mtx;
//    return mtx;
//}

#define PROXY_OPERATOR(enum_name, T)                                                  \
    MUDA_INLINE MUDA_DEVICE LoggerViewer::Proxy& LoggerViewer::Proxy::operator<<(T i) \
    {                                                                                 \
        details::LoggerMetaData meta;                                                 \
        meta.type = details::LoggerBasicType::enum_name;                              \
        meta.size = sizeof(T);                                                        \
        meta.id   = m_log_id;                                                         \
        m_viewer.push_data(meta, &i);                                                 \
        return *this;                                                                 \
    }

PROXY_OPERATOR(Int8, int8_t);
PROXY_OPERATOR(Int16, int16_t);
PROXY_OPERATOR(Int32, int32_t);
PROXY_OPERATOR(Int64, int64_t);

PROXY_OPERATOR(UInt8, uint8_t);
PROXY_OPERATOR(UInt16, uint16_t);
PROXY_OPERATOR(UInt32, uint32_t);
PROXY_OPERATOR(UInt64, uint64_t);

PROXY_OPERATOR(Float, float);
PROXY_OPERATOR(Double, double);

#undef PROXY_OPERATOR
}  // namespace muda