#pragma once
#include <unordered_map>
#include <vector>
namespace muda
{
class ComputeGraphVarBase;
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
    ComputeGraphVar<T>& create_var(std::string_view name, T init_value);

    template <typename T>
    ComputeGraphVar<T>* find_var(std::string_view name);

  private:
    friend class ComputeGraph;
    friend class ComputeGraphNodeBase;
    std::unordered_map<std::string, ComputeGraphVarBase*> m_vars_map;
    std::vector<ComputeGraphVarBase*>                     m_vars;
};
}  // namespace muda

#include <muda/compute_graph/details/compute_graph_var_manager.inl>