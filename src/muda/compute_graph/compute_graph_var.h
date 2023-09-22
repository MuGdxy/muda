#pragma once
#include <string>
#include <set>
#include <map>
#include <muda/type_traits/type_modifier.h>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
namespace muda
{
class ComputeGraphVarBase
{
    std::string_view m_name;
    ComputeGraph*    m_graph = nullptr;
    VarId            m_var_id;
    bool             m_is_valid;

  public:
    std::string_view name() const { return m_name; }
    VarId            var_id() const { return m_var_id; }
    bool             is_valid() const { return m_is_valid; }

  protected:
    friend class ComputeGraph;

    ComputeGraphVarBase(ComputeGraph* compute_graph, std::string_view name, VarId var_id)
        : m_graph(compute_graph)
        , m_name(name)
        , m_var_id(var_id)
        , m_is_valid(false)
    {
    }

    ComputeGraphVarBase(ComputeGraph* compute_graph, std::string_view name, VarId var_id, bool is_valid)
        : m_graph(compute_graph)
        , m_name(name)
        , m_var_id(var_id)
        , m_is_valid(is_valid)
    {
    }

    virtual ~ComputeGraphVarBase() = default;

    mutable std::set<ClosureId> m_closure_ids;

    void base_update();

    void base_building_eval();

    void base_building_eval_const() const;

  private:
    void _building_eval(ComputeGraphVarUsage usage) const;
};

template <typename T>
class ComputeGraphVar : public ComputeGraphVarBase
{
  public:
    using ROViewer = read_only_viewer_t<T>;
    using RWViewer = T;

  protected:
    friend class ComputeGraph;

    using ComputeGraphVarBase::ComputeGraphVarBase;

    ComputeGraphVar(ComputeGraph* compute_graph, std::string_view name, VarId var_id)
        : ComputeGraphVarBase(compute_graph, name, var_id)
    {
    }

    ComputeGraphVar(ComputeGraph* compute_graph, std::string_view name, VarId var_id, T init_value)
        : ComputeGraphVarBase(compute_graph, name, var_id, true)
        , m_value(init_value)
    {
    }

    virtual ~ComputeGraphVar() = default;

  public:
    RWViewer eval();

    ROViewer ceval() const;

    void update(const RWViewer& view);

  private:
    RWViewer m_value;
};
}  // namespace muda


#include <muda/compute_graph/details/compute_graph_var.inl>