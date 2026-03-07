namespace muda
{
template <typename T>
MUDA_INLINE void ComputeGraphVar<VarView<T>>::update(const RWView& view)
{
    ComputeGraphVarBase::update();
    m_value = view;
}
template <typename T>
MUDA_INLINE auto ComputeGraphVar<VarView<T>>::operator=(const RWView& view) -> ComputeGraphVar<VarType>&
{
    update(view);
    return *this;
}
}  // namespace muda