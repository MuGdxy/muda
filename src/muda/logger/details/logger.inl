#include <algorithm>
#include <sstream>
#include <muda/mstl/span.h>
namespace muda
{
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
            put(ss, meta_data);
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

MUDA_INLINE void Logger::put(std::ostream& os, const details::LoggerMetaData& meta_data) const
{
    auto buffer = m_h_buffer.data();
    auto offset = meta_data.offset;
    auto type   = meta_data.type;
#define MUDA_PUT_CASE(EnumT, T)                                                \
    case details::LoggerBasicType::EnumT:                                      \
        os << *reinterpret_cast<const T*>(buffer + offset);                    \
        break;

    switch(type)
    {
        case details::LoggerBasicType::String:
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

MUDA_INLINE Logger::~Logger()
{
    checkCudaErrors(cudaFree(m_buffer));
    checkCudaErrors(cudaFree(m_meta_data));
    checkCudaErrors(cudaFree(m_offset));
}
}  // namespace muda