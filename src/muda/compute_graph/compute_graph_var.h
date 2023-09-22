#pragma once
#include <string>
#include <set>
#include <map>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
namespace muda
{
class ComputeGraphVarBase
{
    std::string_view m_name;
    ComputeGraph*    m_graph = nullptr;
    VarId            m_var_id;

  public:
    std::string_view name() const { return m_name; }
    VarId            var_id() const { return m_var_id; }

  protected:
    friend class ComputeGraph;

    ComputeGraphVarBase(ComputeGraph* compute_graph, std::string_view name, VarId var_id)
        : m_graph(compute_graph)
        , m_name(name)
        , m_var_id(var_id)
    {
    }

    virtual ~ComputeGraphVarBase() = default;

    mutable std::set<ClosureId> m_closure_ids;

    void base_update();

    void base_building_eval();

    void base_building_eval_const() const;
};

template <typename T>
class ComputeGraphVar : public ComputeGraphVarBase
{
  protected:
    friend class ComputeGraph;

    using ComputeGraphVarBase::ComputeGraphVarBase;

    virtual ~ComputeGraphVar() = default;

  public:
    T& eval();

    const T& ceval() const;

    const T& eval() const { return ceval(); }

  private:
    T m_value;
};
}  // namespace muda


#include <muda/compute_graph/details/compute_graph_var.inl>