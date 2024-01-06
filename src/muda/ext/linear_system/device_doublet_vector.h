#pragma once
#pragma once
#include <muda/buffer/device_buffer.h>
#include <Eigen/Core>
#include <muda/ext/linear_system/doublet_vector_viewer.h>

namespace muda::details
{
template <typename T, int N>
class MatrixFormatConverter;
}

namespace muda
{
template <typename T, int N>
class DeviceDoubletVector
{
    template <typename T, int N>
    friend class details::MatrixFormatConverter;

  public:
    using SegmentVector = Eigen::Vector<T, N>;

  protected:
    muda::DeviceBuffer<SegmentVector> m_segment_values;
    muda::DeviceBuffer<int>           m_segment_indices;
    int                               m_segment_count = 0;

  public:
    DeviceDoubletVector()  = default;
    ~DeviceDoubletVector() = default;

    void reshape(int num_segment) { m_segment_count = num_segment; }

    void resize_doublet(size_t nonzero_count)
    {
        m_segment_values.resize(nonzero_count);
        m_segment_indices.resize(nonzero_count);
    }

    void clear()
    {
        m_segment_values.clear();
        m_segment_indices.clear();
    }

    static constexpr int segment_size() { return N; }

    auto segment_count() const { return m_segment_count; }
    auto segment_values() { return m_segment_values.view(); }
    auto segment_values() const { return m_segment_values.view(); }
    auto segment_indices() { return m_segment_indices.view(); }
    auto segment_indices() const { return m_segment_indices.view(); }

    auto doublet_size() const { return m_segment_values.size(); }


    auto viewer()
    {
        return DoubletVectorViewer<T, N>{m_segment_count,
                                         0,
                                         (int)m_segment_values.size(),
                                         (int)m_segment_values.size(),
                                         m_segment_indices.data(),
                                         m_segment_values.data()};
    }
};
}  // namespace muda


#include "details/device_doublet_vector.inl"
