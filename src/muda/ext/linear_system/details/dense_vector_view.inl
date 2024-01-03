namespace muda
{
template <typename T>
CDenseVectorViewer<T> DenseVectorViewBase<T>::cviewer() const
{
    return CDenseVectorViewer<T>{m_view, 0, m_view.size()};
}
template <typename T>
DenseVectorViewer<T> DenseVectorView<T>::viewer()
{
    return DenseVectorViewer{m_view, 0, m_view.size()};
}
}  // namespace muda
