#include <muda/compute_graph/compute_graph_var.h>
#include <numeric>
#include <algorithm>
#include "compute_graph_var_manager.h"

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
MUDA_INLINE ComputeGraphVar<T>& ComputeGraphVarManager::create_var(std::string_view name,
                                                                   const T& init_value)
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
}

template <typename... T>
MUDA_INLINE bool ComputeGraphVarManager::is_using(const ComputeGraphVar<T>&... vars) const
{
    std::array<const ComputeGraphVarBase*, sizeof...(T)> var_array{&vars...};
    return is_using(span<const ComputeGraphVarBase*>{var_array});
}
template <typename... T>
MUDA_INLINE void ComputeGraphVarManager::sync(const ComputeGraphVar<T>&... vars) const
{
    std::array<const ComputeGraphVarBase*, sizeof...(T)> var_array{&vars...};
    sync(span<const ComputeGraphVarBase*>{var_array});
};

MUDA_INLINE bool ComputeGraphVarManager::is_using() const
{
    return is_using(var_span());
}

MUDA_INLINE void ComputeGraphVarManager::sync() const
{
    sync(var_span());
}

MUDA_INLINE bool ComputeGraphVarManager::is_using(const span<const ComputeGraphVarBase*> vars) const
{
    auto graphs = unique_graphs(vars);
    return std::any_of(graphs.begin(),
                       graphs.end(),
                       [](ComputeGraph* graph) {
                           return graph->query() == Event::QueryResult::eNotReady;
                       });
}

MUDA_INLINE void ComputeGraphVarManager::sync(const span<const ComputeGraphVarBase*> vars,
                                              cudaStream_t stream) const
{
    auto graphs = unique_graphs(vars);
    std::for_each(graphs.begin(),
                  graphs.end(),
                  [&](ComputeGraph* graph) { on(stream).wait(graph->m_event); });
}

inline std::vector<ComputeGraph*> ComputeGraphVarManager::graphs() const
{
    return unique_graphs(var_span());
}

MUDA_INLINE std::vector<ComputeGraph*> ComputeGraphVarManager::unique_graphs(
    span<const ComputeGraphVarBase*> vars) const
{
    std::vector<ComputeGraph*> graphs;
    for(auto var : vars)
    {
        for(auto& [graph, _] : var->m_related_closure_infos)
        {
            graphs.emplace_back(graph);
        }
    }
    std::sort(graphs.begin(), graphs.end());
    // get unique graphs
    graphs.erase(std::unique(graphs.begin(), graphs.end()), graphs.end());
    return graphs;
}

MUDA_INLINE span<const ComputeGraphVarBase*> ComputeGraphVarManager::var_span() const
{
    return span<const ComputeGraphVarBase*>{
        const_cast<const ComputeGraphVarBase**>(m_vars.data()), m_vars.size()};
}
}  // namespace muda
