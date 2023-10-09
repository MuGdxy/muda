#include <muda/compute_graph/compute_graph_var.h>

namespace muda
{
MUDA_INLINE ComputeGraphVarManager::~ComputeGraphVarManager()
{
    for(auto& var : m_vars)
        delete var;
}

template <typename T>
MUDA_INLINE ComputeGraphVar<T>& ComputeGraphVarManager::create_var(std::string_view name)
{
    auto ptr = new ComputeGraphVar<T>(this, name, VarId{m_vars.size()});
    m_vars.emplace_back(ptr);
    if(m_vars_map.find(std::string{name}) != m_vars_map.end())
        MUDA_ERROR_WITH_LOCATION("var[%s] already exists", name.data());
    m_vars_map.emplace(name, ptr);
    return *ptr;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<T>& ComputeGraphVarManager::create_var(std::string_view name, T init_value)
{
    auto ptr = new ComputeGraphVar<T>(this, name, VarId{m_vars.size()}, init_value);
    m_vars.emplace_back(ptr);
    m_vars_map.emplace(name, ptr);
    return *ptr;
}
template <typename T>
MUDA_INLINE ComputeGraphVar<T>* ComputeGraphVarManager::find_var(std::string_view name)
{
    auto it = m_vars_map.find(std::string{name});
    if(it == m_vars_map.end())
        return nullptr;
    return dynamic_cast<ComputeGraphVar<T>*>(it->second);
};
}  // namespace muda
