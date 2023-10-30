#pragma once
#include <unordered_map>
#include <vector>
#include <muda/mstl/span.h>
namespace muda
{
class ComputeGraphVarBase;
class ComputeGraph;
class ComputeGraphGraphvizOptions;

class ComputeGraphVarManager
{
  public:
    ComputeGraphVarManager() = default;
    ~ComputeGraphVarManager();
    /**************************************************************
    * 
    * GraphVar API
    * 
    ***************************************************************/
    template <typename T>
    ComputeGraphVar<T>& create_var(std::string_view name);
    template <typename T>
    ComputeGraphVar<T>& create_var(std::string_view name, const T& init_value);
    template <typename T>
    ComputeGraphVar<T>* find_var(std::string_view name);

    bool is_using() const;
    void sync() const;

    template <typename... T>
    bool is_using(const ComputeGraphVar<T>&... vars) const;
    template <typename... T>
    void sync(const ComputeGraphVar<T>&... vars) const;

    bool is_using(const span<const ComputeGraphVarBase*> vars) const;
    void sync(const span<const ComputeGraphVarBase*> vars, cudaStream_t stream = nullptr) const;

    const auto& graphs() const { return m_graphs; }
    void graphviz(std::ostream& os, const ComputeGraphGraphvizOptions& options = {}) const;

  private:
    friend class ComputeGraph;
    friend class ComputeGraphNodeBase;
    friend class ComputeGraphClosure;
    std::vector<ComputeGraph*> unique_graphs(span<const ComputeGraphVarBase*> vars) const;
    std::unordered_map<std::string, ComputeGraphVarBase*> m_vars_map;
    std::vector<ComputeGraphVarBase*>                     m_vars;
    std::unordered_set<ComputeGraph*>                     m_graphs;
    span<const ComputeGraphVarBase*>                      var_span() const;
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_var_manager.inl>