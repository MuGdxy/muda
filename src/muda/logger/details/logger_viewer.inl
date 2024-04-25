#include <muda/atomic.h>
namespace muda
{
MUDA_INLINE MUDA_DEVICE LogProxy::LogProxy(LoggerViewer& viewer)
    : m_viewer(&viewer)
{
    MUDA_KERNEL_ASSERT(m_viewer->m_buffer && m_viewer->m_meta_data,
                       "LoggerViewer is not initialized");
    m_log_id = atomic_add(&(m_viewer->m_offset->log_id), 1u);
}
template <bool IsFmt>
MUDA_INLINE MUDA_DEVICE LogProxy& LogProxy::push_string(const char* str)
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
    if constexpr(IsFmt)
    {
        meta.type = LoggerBasicType::FmtString;
    }
    else
    {
        meta.type = LoggerBasicType::String;
    }
    meta.size = size;
    meta.id   = m_log_id;
    m_viewer->push_data(meta, str);
    return *this;
}

template <typename T>
MUDA_DEVICE void LogProxy::push_fmt_arg(const T& obj, LoggerFmtArg func)
{
    details::LoggerMetaData meta;
    meta.type    = LoggerBasicType::Object;
    meta.size    = sizeof(T);
    meta.id      = m_log_id;
    meta.fmt_arg = func;
    m_viewer->push_data(meta, &obj);
}

MUDA_INLINE MUDA_DEVICE LogProxy& LogProxy::operator<<(const char* str)
{
    return push_string<false>(str);
}

template <typename T>
MUDA_INLINE MUDA_DEVICE LogProxy& LoggerViewer::operator<<(const T& t)
{
    m_proxy = LogProxy(*this);
    m_proxy << t;
    return m_proxy;
}

template <bool IsFmt>
MUDA_INLINE MUDA_DEVICE LogProxy& LoggerViewer::push_string(const char* str)
{
    m_proxy = LogProxy(*this);
    m_proxy.push_string<IsFmt>(str);
    return m_proxy;
}

MUDA_INLINE MUDA_DEVICE LogProxy& LoggerViewer::operator<<(const char* s)
{
    m_proxy = LogProxy(*this);
    m_proxy << s;
    return m_proxy;
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
        old             = atomic_cas(data_offset, assumed, new_offset);
        if(old + size >= total_size)
        {
            old = ~0u;
            break;
        }
    } while(assumed != old);
    return old;
}

MUDA_INLINE MUDA_DEVICE uint32_t LoggerViewer::next_meta_data_idx() const
{
    auto idx = next_idx(&(m_offset->meta_data_offset), 1u, m_meta_data_size);
    if(idx == ~0u)
    {
        atomic_cas(&(m_offset->exceed_meta_data), 0u, 1u);
        return ~0u;
    }
    return idx;
}

MUDA_INLINE MUDA_DEVICE uint32_t LoggerViewer::next_buffer_idx(uint32_t size) const
{
    auto idx = next_idx(&(m_offset->buffer_offset), size, m_buffer_size);
    if(idx == ~0u)
    {
        atomic_cas(&(m_offset->exceed_buffer), 0u, 1u);
        return ~0u;
    }
    return idx;
}

MUDA_INLINE MUDA_DEVICE bool LoggerViewer::push_data(details::LoggerMetaData meta,
                                                     const void* data)
{
    auto meta_idx = next_meta_data_idx();
    if(meta_idx == ~0u)
    {
        MUDA_KERNEL_WARN_WITH_LOCATION(
            "LoggerViewer: meta data is exceeded, "
            "the content[id=%d] will be discarded.",
            meta.id);
        return false;
    }
    auto buffer_idx = next_buffer_idx(meta.size);
    if(buffer_idx == ~0u)
    {
        meta.exceeded = true;
        MUDA_KERNEL_WARN_WITH_LOCATION(
            "LoggerViewer: log buffer is exceeded, "
            "the content[id=%d] will be discarded.",
            meta.id);

        m_meta_data[meta_idx]    = meta;
        m_meta_data_id[meta_idx] = meta.id;
        return false;
    }
    meta.offset              = buffer_idx;
    m_meta_data[meta_idx]    = meta;
    m_meta_data_id[meta_idx] = meta.id;
    for(int i = 0; i < meta.size; ++i)
        m_buffer[buffer_idx + i] = reinterpret_cast<const char*>(data)[i];
    return true;
}

#define PROXY_OPERATOR(enum_name, T)                                           \
    MUDA_INLINE MUDA_DEVICE LogProxy& LogProxy::operator<<(T i)                \
    {                                                                          \
        details::LoggerMetaData meta;                                          \
        meta.type = LoggerBasicType::enum_name;                                \
        meta.size = sizeof(T);                                                 \
        meta.id   = m_log_id;                                                  \
        m_viewer->push_data(meta, &i);                                         \
        return *this;                                                          \
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

#ifdef __linux__
MUDA_INLINE MUDA_DEVICE LogProxy& LogProxy::operator<<(long long ll)
{
    details::LoggerMetaData meta;
    meta.type = LoggerBasicType::Int64;
    meta.size = sizeof(long long);
    meta.id   = m_log_id;
    m_viewer->push_data(meta, &ll);
    return *this;
}

MUDA_INLINE MUDA_DEVICE LogProxy& LogProxy::operator<<(unsigned long long ull)
{
    details::LoggerMetaData meta;
    meta.type = LoggerBasicType::UInt64;
    meta.size = sizeof(unsigned long long);
    meta.id   = m_log_id;
    m_viewer->push_data(meta, &ull);
    return *this;
}
#endif

#undef PROXY_OPERATOR
}  // namespace muda
