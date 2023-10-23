#pragma once
#include <muda/muda_def.h>
#include <muda/check/check_cuda_errors.h>
#include <cinttypes>
#include <muda/buffer/device_buffer.h>
#include <muda/literal/unit.h>
#include <muda/viewer/dense.h>
#include <mutex>

namespace muda
{
namespace details
{
    enum class LoggerBasicType : uint16_t
    {
        Int,
        UInt,
        Int64,
        UInt64,
        Float,
        Double,
        String,
    };

    class LoggerMetaData
    {
      public:
        LoggerBasicType type;
        uint16_t        exceeded = 0;  // false
        uint32_t        id;
        uint32_t        size;
        uint32_t        offset;

        void put(std::ostream& os, char* buffer) const;
    };

    class LoggerOffset
    {
      public:
        uint32_t log_id           = 0;
        uint32_t meta_data_offset = 0;
        uint32_t exceed_meta_data = 0;  // false
        uint32_t buffer_offset    = 0;
        uint32_t exceed_buffer    = 0;  // false
    };
}  // namespace details


class LoggerViewer
{
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

        MUDA_DEVICE Proxy& operator<<(const char* str);
        MUDA_DEVICE Proxy& operator<<(int i);
        MUDA_DEVICE Proxy& operator<<(uint32_t i);
        MUDA_DEVICE Proxy& operator<<(int64_t i);
        MUDA_DEVICE Proxy& operator<<(uint64_t i);
        MUDA_DEVICE Proxy& operator<<(float f);
        MUDA_DEVICE Proxy& operator<<(double d);
    };

    friend class Logger;

    template <typename T>
    MUDA_DEVICE Proxy operator<<(const T& t);
    MUDA_DEVICE Proxy operator<<(const char* s);

  public:
    details::LoggerMetaData*       m_meta_data_view_data;
    uint32_t                       m_meta_data_view_size;
    char*                          m_buffer_view_data;
    uint32_t                       m_buffer_view_size;
    mutable details::LoggerOffset* m_offset_view;

    MUDA_DEVICE uint32_t next_meta_data_idx() const;
    MUDA_DEVICE uint32_t next_buffer_idx(uint32_t size) const;
    MUDA_DEVICE bool push_data(details::LoggerMetaData meta, const void* data);
};

class Logger
{
  public:
    Logger(LoggerViewer* symbol = nullptr, size_t meta_size = 16_M, size_t buffer_size = 128_M);
    ~Logger();

    void         retrieve(std::ostream&);
    LoggerViewer viewer() const { return m_viewer; }

  private:
    friend class LaunchCore;
    friend class Debug;
    void expand_meta_data();
    void expand_buffer();
    void upload();
    void download();
    void expand_if_needed();

    // DeviceBuffer<details::LoggerMetaData> m_meta_data;
    details::LoggerMetaData*             m_meta_data;
    size_t                               m_meta_data_size;
    std::vector<details::LoggerMetaData> m_h_meta_data;

    // DeviceBuffer<char> m_buffer;
    char*             m_buffer;
    size_t            m_buffer_size;
    std::vector<char> m_h_buffer;

    // DeviceVar<details::LoggerOffset> m_offset;
    details::LoggerOffset* m_offset;
    details::LoggerOffset  m_h_offset;

    LoggerViewer* m_log_viewer_ptr;
    LoggerViewer  m_viewer;

    static Logger&     instance();
    static std::mutex& mutex();
};

MUDA_INLINE __device__ LoggerViewer cout;
}  // namespace muda

#include <muda/logger/details/logger.inl>