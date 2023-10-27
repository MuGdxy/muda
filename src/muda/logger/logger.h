#pragma once
#include <muda/muda_def.h>
#include <muda/check/check_cuda_errors.h>
#include <cinttypes>
#include <muda/literal/unit.h>
#include <muda/logger/logger_viewer.h>
#include <vector>

namespace muda
{
class Logger
{
    static constexpr size_t DEFAULT_META_SIZE   = 16_M;
    static constexpr size_t DEFAULT_BUFFER_SIZE = 128_M;

  public:
    Logger(LoggerViewer* global_viewer,
           size_t        meta_size   = DEFAULT_META_SIZE,
           size_t        buffer_size = DEFAULT_BUFFER_SIZE);

    Logger(size_t meta_size = DEFAULT_META_SIZE, size_t buffer_size = DEFAULT_BUFFER_SIZE)
        : Logger(nullptr, meta_size, buffer_size)
    {
    }

    ~Logger();

    void         retrieve(std::ostream&);
    LoggerViewer viewer() const { return m_log_viewer_ptr ? *m_log_viewer_ptr : m_viewer; }

  private:
    friend class LaunchCore;
    friend class Debug;
    void expand_meta_data();
    void expand_buffer();
    void upload();
    void download();
    void expand_if_needed();

    details::LoggerMetaData*             m_meta_data;
    size_t                               m_meta_data_size;
    std::vector<details::LoggerMetaData> m_h_meta_data;

    char*             m_buffer;
    size_t            m_buffer_size;
    std::vector<char> m_h_buffer;

    details::LoggerOffset* m_offset;
    details::LoggerOffset  m_h_offset;

    LoggerViewer* m_log_viewer_ptr;
    LoggerViewer  m_viewer;

    void put(std::ostream& os, const details::LoggerMetaData& meta_data) const;
};

//MUDA_INLINE __device__ LoggerViewer cout;
}  // namespace muda

#include <muda/logger/details/logger.inl>