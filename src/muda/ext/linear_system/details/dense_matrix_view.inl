namespace muda
{
template <typename Ty>
auto DenseMatrixViewBase<Ty>::T() const -> DenseMatrixViewBase<value_type>
{
    return DenseMatrixViewBase<value_type>{m_view, m_row, m_col, !m_trans, m_sym};
}
template <typename Ty>
CDenseMatrixViewer<Ty> DenseMatrixViewBase<Ty>::cviewer() const
{
    MUDA_ASSERT(!m_trans,
                "DenseMatrixViewer doesn't support transpose, "
                "please use the original matrix to create a viewer");
    return CDenseMatrixViewer<value_type>{m_view, 0, 0, m_row, m_col};
}

template <typename Ty>
auto DenseMatrixView<Ty>::viewer() -> DenseMatrixViewer<value_type>
{
    MUDA_ASSERT(!m_trans,
                "DenseMatrixViewer doesn't support transpose, "
                "please use the original matrix to create a viewer");
    return DenseMatrixViewer<value_type>{m_view, 0, 0, m_row, m_col};
}
}  // namespace muda