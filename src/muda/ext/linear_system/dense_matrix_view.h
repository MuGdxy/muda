#pragma once
#include <muda/ext/linear_system/dense_matrix_viewer.h>
#include <muda/buffer/buffer_2d_view.h>
namespace muda
{
template <typename Ty>
class DenseMatrixViewBase
{
    static_assert(std::is_same_v<Ty, float> || std::is_same_v<Ty, double>,
                  "now only support real number");

  protected:
    Buffer2DView<Ty> m_view;
    size_t           m_row   = 0;
    size_t           m_col   = 0;
    bool             m_trans = false;
    bool             m_sym   = false;

  public:
    using value_type = Ty;
    DenseMatrixViewBase(const Buffer2DView<Ty>& view, size_t row, size_t col, bool trans, bool sym)
        : m_view(view)
        , m_row(row)
        , m_col(col)
        , m_trans(trans)
        , m_sym(sym)
    {
    }

    bool   is_trans() const { return m_trans; }
    bool   is_sym() const { return m_sym; }
    size_t row() const { return m_row; }
    size_t col() const { return m_col; }
    size_t lda() const { return m_view.pitch_bytes() / sizeof(value_type); }
    DenseMatrixViewBase<value_type> T() const;
    auto                   data() const { return m_view.origin_data(); }
    CDenseMatrixViewer<Ty> cviewer() const;
};

template <typename Ty>
class CDenseMatrixView : public DenseMatrixViewBase<Ty>
{
    using Base = DenseMatrixViewBase<Ty>;

  public:
    using value_type = Ty;
    CDenseMatrixView(const CBuffer2DView<value_type>& view, size_t row, size_t col, bool trans, bool sym)
        : Base(Buffer2DViewBase<value_type>{view}, row, col, trans, sym)
    {
    }

    CDenseMatrixView(const Base& base)
        : Base(base)
    {
    }

    CDenseMatrixView<Ty> T() const { return Base::T(); }
};

template <typename Ty>
class DenseMatrixView : public DenseMatrixViewBase<Ty>
{
    using Base = DenseMatrixViewBase<Ty>;

  public:
    using value_type = Ty;
    DenseMatrixView(const Base& base)
        : Base(base)
    {
    }

    DenseMatrixView(const CDenseMatrixView<value_type>&) = delete;

    auto T() const { return Base::T(); }

    DenseMatrixViewer<Ty> viewer();

    using Base::data;
    auto data() { return m_view.origin_data(); }
};
}  // namespace muda

#include "details/dense_matrix_view.inl"
