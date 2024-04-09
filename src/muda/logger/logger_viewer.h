#pragma once
#include <muda/logger/logger_basic_data.h>
#include <muda/muda_def.h>
#include <muda/check/check_cuda_errors.h>
#include <muda/literal/unit.h>
#include <muda/viewer/dense.h>
namespace muda
{
class LoggerViewer;
class LogProxy
{
    LoggerViewer& m_viewer;
    uint32_t      m_log_id;

  public:
    MUDA_DEVICE LogProxy(LoggerViewer& viewer);

    MUDA_DEVICE LogProxy(const LogProxy& other)
        : m_viewer(other.m_viewer)
        , m_log_id(other.m_log_id)
    {
    }

    template <bool IsFmt>
    MUDA_DEVICE LogProxy& push_string(const char* str);

    MUDA_DEVICE LogProxy& operator<<(const char* str);

    MUDA_DEVICE LogProxy& operator<<(int8_t i);
    MUDA_DEVICE LogProxy& operator<<(int16_t i);
    MUDA_DEVICE LogProxy& operator<<(int32_t i);
    MUDA_DEVICE LogProxy& operator<<(int64_t i);

    MUDA_DEVICE LogProxy& operator<<(uint8_t i);
    MUDA_DEVICE LogProxy& operator<<(uint16_t i);
    MUDA_DEVICE LogProxy& operator<<(uint32_t i);
    MUDA_DEVICE LogProxy& operator<<(uint64_t i);


    MUDA_DEVICE LogProxy& operator<<(float f);
    MUDA_DEVICE LogProxy& operator<<(double d);

    template <typename T>
    MUDA_DEVICE void push_fmt_arg(const T& obj, LoggerFmtArg fmt_arg_func);
};
class LoggerViewer
{
  public:
    friend class Logger;

    template <typename T>
    MUDA_DEVICE LogProxy operator<<(const T& t);
    MUDA_DEVICE LogProxy operator<<(const char* s);
    template <bool IsFmt>
    MUDA_DEVICE LogProxy push_string(const char* str);
    MUDA_DEVICE LogProxy proxy() { return LogProxy(*this); }

  public:
    // Don't use viewer, cuda don't allow to use constructor in __device__ global variable
    // However, LoggerViewer should be able to use as a global variable for debugging
    uint32_t*                m_meta_data_id      = nullptr;
    int                      m_meta_data_id_size = 0;
    details::LoggerMetaData* m_meta_data         = nullptr;
    int                      m_meta_data_size    = 0;
    char*                    m_buffer            = nullptr;
    int                      m_buffer_size       = 0;
    details::LoggerOffset*   m_offset            = nullptr;

    MUDA_DEVICE uint32_t next_meta_data_idx() const;
    MUDA_DEVICE uint32_t next_buffer_idx(uint32_t size) const;
    MUDA_DEVICE bool push_data(details::LoggerMetaData meta, const void* data);
};
}  // namespace muda

#include <muda/logger/details/logger_viewer.inl>