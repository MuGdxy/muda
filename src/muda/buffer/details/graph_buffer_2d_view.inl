namespace muda
{
template <typename T>
MUDA_INLINE void ComputeGraphVar<Buffer2DView<T>>::update(const RWView& view)
{
    ComputeGraphVarBase::update();
    m_value = view;
}
template <typename T>
MUDA_INLINE auto ComputeGraphVar<Buffer2DView<T>>::operator=(const RWView& view) -> ComputeGraphVar<VarType>&
{
    update(view);
    return *this;
}
}  // namespace muda