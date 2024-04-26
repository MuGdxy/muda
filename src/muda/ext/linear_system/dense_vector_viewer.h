#pragma once
#include <Eigen/Core>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/viewer/viewer_base.h>
#include <muda/viewer/viewer_base_accessor.h>
#include <cublas_v2.h>
#include <muda/atomic.h>
namespace muda
{
template <bool IsConst, typename T>
class DenseVectorViewerBase : public ViewerBase<IsConst>
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "now only support real number");

    using Base = ViewerBase<IsConst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;

  public:
    using CBufferView    = CBufferView<T>;
    using BufferView     = BufferView<T>;
    using ThisBufferView = std::conditional_t<IsConst, CBufferView, BufferView>;

    using ConstViewer    = DenseVectorViewerBase<true, T>;
    using NonConstViewer = DenseVectorViewerBase<false, T>;
    using ThisViewer = std::conditional_t<IsConst, ConstViewer, NonConstViewer>;

    using VectorType = Eigen::Vector<T, Eigen::Dynamic>;
    template <typename U>
    using MapVectorT =
        Eigen::Map<U, Eigen::AlignmentType::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MapVector     = MapVectorT<VectorType>;
    using CMapVector    = MapVectorT<const VectorType>;
    using ThisMapVector = std::conditional_t<IsConst, CMapVector, MapVector>;

  protected:
    auto_const_t<T>* m_data;
    int              m_offset      = 0;
    int              m_size        = 0;
    int              m_origin_size = 0;

  public:
    MUDA_GENERIC DenseVectorViewerBase(auto_const_t<T>* data, int offset, int size, int origin_size)
        : m_data(data)
        , m_offset(offset)
        , m_size(size)
        , m_origin_size(origin_size)
    {
    }

    MUDA_GENERIC auto as_const() const
    {
        return ConstViewer{m_data, m_offset, m_size, m_origin_size};
    }

    MUDA_GENERIC operator ConstViewer() const { return as_const(); }

    MUDA_GENERIC auto segment(int offset, int size)
    {
        check_segment(offset, size);
        auto ret = ThisViewer{m_data, m_offset + offset, size, m_origin_size};
        ret.copy_name(*this);
        return ret;
    }

    template <int N>
    MUDA_GENERIC auto segment(int offset)
    {
        return segment(offset, N);
    }

    MUDA_GENERIC auto segment(int offset, int size) const
    {
        return remove_const(*this).segment(offset, size);
    }

    MUDA_GENERIC const T& operator()(int i) const { return m_data[index(i)]; }
    MUDA_GENERIC auto_const_t<T>& operator()(int i) { return m_data[index(i)]; }

    template <int N>
    MUDA_GENERIC auto segment(int offset) const
    {
        return remove_const(*this).segment(offset, N);
    }

    MUDA_GENERIC Eigen::VectorBlock<CMapVector> as_eigen() const
    {
        check_data();
        return CMapVector{m_data,
                          (int)origin_size(),
                          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1}}
            .segment(m_offset, m_size);
    }

    MUDA_GENERIC operator Eigen::VectorBlock<CMapVector>() const
    {
        return as_eigen();
    }

    MUDA_GENERIC Eigen::VectorBlock<ThisMapVector> as_eigen()
    {
        check_data();
        return ThisMapVector{m_data,
                             (int)origin_size(),
                             Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>{1, 1}}
            .segment(m_offset, m_size);
    }

    MUDA_GENERIC operator Eigen::VectorBlock<ThisMapVector>()
    {
        return as_eigen();
    }

    MUDA_GENERIC auto size() const { return m_size; }
    MUDA_GENERIC auto offset() const { return m_offset; }
    MUDA_GENERIC auto origin_data() const { return m_data; }
    MUDA_GENERIC auto origin_size() const { return m_origin_size; }


  protected:
    MUDA_INLINE MUDA_GENERIC void check_size_matching(int N)
    {
        MUDA_KERNEL_ASSERT(m_size == N,
                           "DenseVectorViewerBase [%s:%s]: size not match, yours size=%d, expected size=%d",
                           this->name(),
                           this->kernel_name(),
                           m_size,
                           N);
    }

    MUDA_INLINE MUDA_GENERIC int index(int i) const
    {
        MUDA_KERNEL_ASSERT(origin_data(),
                           "DenseVectorViewerBase [%s:%s]: data is null",
                           this->name(),
                           this->kernel_name());
        MUDA_KERNEL_ASSERT(i < m_size,
                           "DenseVectorViewerBase [%s:%s]: index out of range, size=%d, yours index=%d",
                           this->name(),
                           this->kernel_name(),
                           m_size,
                           i);
        return m_offset + i;
    }

    MUDA_INLINE MUDA_GENERIC void check_data() const
    {
        MUDA_KERNEL_ASSERT(origin_data(),
                           "DenseVectorViewerBase [%s:%s]: data is null",
                           this->name(),
                           this->kernel_name());
    }

    MUDA_INLINE MUDA_GENERIC void check_segment(int offset, int size) const
    {
        MUDA_KERNEL_ASSERT(offset + size <= m_size,
                           "DenseVectorViewerBase [%s:%s]: segment out of range, m_size=%d, offset=%d, size=%d",
                           this->name(),
                           this->kernel_name(),
                           m_size,
                           offset,
                           size);
    }
};

//template <typename T>
//using CDenseVectorViewer = DenseVectorViewerBase<true, T>;
//template <typename T>
//using DenseVectorViewer = DenseVectorViewerBase<false, T>;

template <typename T>
class CDenseVectorViewer : public DenseVectorViewerBase<true, T>
{
    MUDA_VIEWER_COMMON_NAME(CDenseVectorViewer);

    using Base       = DenseVectorViewerBase<true, T>;
    using MapVector  = typename Base::MapVector;
    using CMapVector = typename Base::CMapVector;

  public:
    using Base::Base;

    MUDA_GENERIC CDenseVectorViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC CDenseVectorViewer segment(int offset, int size) const
    {
        return CDenseVectorViewer{Base::segment(offset, size)};
    }

    template <int N>
    MUDA_GENERIC auto segment(int offset) const
    {
        return segment(offset, N);
    }
};

template <typename T>
class DenseVectorViewer : public DenseVectorViewerBase<false, T>
{
    MUDA_VIEWER_COMMON_NAME(DenseVectorViewer);

    using Base       = DenseVectorViewerBase<false, T>;
    using MapVector  = typename Base::MapVector;
    using CMapVector = typename Base::CMapVector;

  public:
    using Base::Base;

    MUDA_GENERIC DenseVectorViewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC DenseVectorViewer segment(int offset, int size)
    {
        return DenseVectorViewer{Base::segment(offset, size)};
    }

    template <int N>
    MUDA_GENERIC auto segment(int offset)
    {
        return segment(offset, N);
    }

    MUDA_DEVICE T atomic_add(int i, T val)
    {
        auto ptr = &this->operator()(i);
        return muda::atomic_add(ptr, val);
    }

    template <int N>
    MUDA_DEVICE Eigen::Vector<T, N> atomic_add(const Eigen::Vector<T, N>& val)
    {
        check_size_matching(N);
        Eigen::Vector<T, N> ret;
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            ret(i) = atomic_add(i, val(i));
        }
        return ret;
    }

    MUDA_DEVICE T atomic_add(const T& val)
    {
        this->check_size_matching(1);
        T ret = atomic_add(0, val);
        return ret;
    }

    template <int N>
    MUDA_GENERIC DenseVectorViewer& operator=(const Eigen::Vector<T, N>& other)
    {
        this->check_size_matching(N);
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            this->operator()(i) = other(i);
        }
        return *this;
    }
};
}  // namespace muda

namespace muda
{
template <typename T>
struct read_only_viewer<DenseVectorViewer<T>>
{
    using type = CDenseVectorViewer<T>;
};

template <typename T>
struct read_write_viewer<CDenseVectorViewer<T>>
{
    using type = DenseVectorViewer<T>;
};
}  // namespace muda

#include "details/dense_vector_viewer.inl"