#pragma once
#include <muda/logger/logger_basic_data.h>
#include <muda/muda_def.h>
#include <muda/check/check_cuda_errors.h>
#include <muda/literal/unit.h>
#include <muda/viewer/dense.h>
namespace muda
{
class LoggerViewer : public ViewerBase
{
    MUDA_VIEWER_COMMON_NAME(LoggerViewer);

  public:
    class Proxy
    {
        LoggerViewer& m_viewer;
        uint32_t      m_log_id;

      public:
        MUDA_DEVICE Proxy(LoggerViewer& viewer);

        MUDA_DEVICE Proxy(const Proxy& other)
            : m_viewer(other.m_viewer)
            , m_log_id(other.m_log_id)
        {
        }

        template <bool IsFmt>
        MUDA_DEVICE Proxy& push_string(const char* str);

        MUDA_DEVICE Proxy& operator<<(const char* str);

        MUDA_DEVICE Proxy& operator<<(int8_t i);
        MUDA_DEVICE Proxy& operator<<(int16_t i);
        MUDA_DEVICE Proxy& operator<<(int32_t i);
        MUDA_DEVICE Proxy& operator<<(int64_t i);

        MUDA_DEVICE Proxy& operator<<(uint8_t i);
        MUDA_DEVICE Proxy& operator<<(uint16_t i);
        MUDA_DEVICE Proxy& operator<<(uint32_t i);
        MUDA_DEVICE Proxy& operator<<(uint64_t i);


        MUDA_DEVICE Proxy& operator<<(float f);
        MUDA_DEVICE Proxy& operator<<(double d);

        template <typename T>
        MUDA_DEVICE void push_fmt_arg(const T& obj, LoggerFmtArg fmt_arg_func);
    };

    friend class Logger;

    template <typename T>
    MUDA_DEVICE Proxy operator<<(const T& t);
    MUDA_DEVICE Proxy operator<<(const char* s);
    template <bool IsFmt>
    MUDA_DEVICE Proxy push_string(const char* str);
    MUDA_DEVICE Proxy proxy() { return Proxy(*this); }

  private:
    Dense1D<uint32_t>                    m_meta_data_id;
    Dense1D<details::LoggerMetaData>     m_meta_data;
    Dense1D<char>                        m_buffer;
    mutable Dense<details::LoggerOffset> m_offset_view;

    MUDA_DEVICE uint32_t next_meta_data_idx() const;
    MUDA_DEVICE uint32_t next_buffer_idx(uint32_t size) const;
    MUDA_DEVICE bool push_data(details::LoggerMetaData meta, const void* data);
};

using LogProxy = LoggerViewer::Proxy;
}  // namespace muda

#include <muda/logger/details/logger_viewer.inl>