namespace muda
{
template <bool IsConst, typename Ty>
MUDA_GENERIC auto DenseMatrixViewBase<IsConst, Ty>::T() MUDA_NOEXCEPT->ThisView
{
    return ThisView{m_view, m_row, m_col, !m_trans, m_sym};
}

template <bool IsConst, typename Ty>
auto DenseMatrixViewBase<IsConst, Ty>::viewer() MUDA_NOEXCEPT->ThisViewer
{
    MUDA_ASSERT(!m_trans,
                "DenseMatrixViewer doesn't support transpose, "
                "please use the original matrix to create a viewer");
    return ThisViewer{m_view, 0, 0, m_row, m_col};
}
}  // namespace muda