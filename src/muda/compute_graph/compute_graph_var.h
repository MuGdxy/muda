#pragma once
#include <string>
#include <set>
#include <map>
#include <muda/type_traits/type_modifier.h>
#include <muda/compute_graph/compute_graph_closure_id.h>
#include <muda/compute_graph/compute_graph_var_usage.h>
namespace muda
{
class ComputeGraph;

class ComputeGraphVarManager;

class ComputeGraphVarBase
{
    std::string_view        m_name;
    ComputeGraphVarManager* m_var_manager = nullptr;
    VarId                   m_var_id;
    bool                    m_is_valid;

  public:
    std::string_view name() const { return m_name; }
    VarId            var_id() const { return m_var_id; }
    bool             is_valid() const { return m_is_valid; }
    void             graphviz_def(std::ostream& os) const;
    void             graphviz_id(std::ostream& os) const;

  protected:
    friend class ComputeGraph;

    ComputeGraphVarBase(ComputeGraphVarManager* var_manager, std::string_view name, VarId var_id)
        : m_var_manager(var_manager)
        , m_name(name)
        , m_var_id(var_id)
        , m_is_valid(false)
    {
    }

    ComputeGraphVarBase(ComputeGraphVarManager* var_manager, std::string_view name, VarId var_id, bool is_valid)
        : m_var_manager(var_manager)
        , m_name(name)
        , m_var_id(var_id)
        , m_is_valid(is_valid)
    {
    }

    friend class ComputeGraphVarManager;
    virtual ~ComputeGraphVarBase() = default;

    mutable std::set<ClosureId> m_closure_ids;

    void base_update();

    void base_building_eval();

    void base_building_eval_const() const;

  private:
    void _building_eval(ComputeGraphVarUsage usage) const;

    class RelatedClosureInfo
    {
      public:
        ComputeGraph*       graph;
        std::set<ClosureId> closure_ids;
    };

    mutable std::map<ComputeGraph*, RelatedClosureInfo> m_related_closure_infos;

    void remove_related_closure_infos(ComputeGraph* graph);
};

template <typename T>
class ComputeGraphVar : public ComputeGraphVarBase
{
  public:
    using ROViewer = read_only_viewer_t<T>;
    using RWViewer = T;

  protected:
    friend class ComputeGraph;
    friend class ComputeGraphVarManager;

    using ComputeGraphVarBase::ComputeGraphVarBase;

    ComputeGraphVar(ComputeGraphVarManager* var_manager, std::string_view name, VarId var_id)
        : ComputeGraphVarBase(var_manager, name, var_id)
    {
    }

    ComputeGraphVar(ComputeGraphVarManager* var_manager, std::string_view name, VarId var_id, T init_value)
        : ComputeGraphVarBase(var_manager, name, var_id, true)
        , m_value(init_value)
    {
    }

    virtual ~ComputeGraphVar() = default;

  public:
    RWViewer eval();

    ROViewer ceval() const;

    void update(const RWViewer& view);

    operator ROViewer() const { return ceval(); }

    operator RWViewer() { return eval(); }

  private:
    RWViewer m_value;
};
}  // namespace muda


#include <muda/compute_graph/details/compute_graph_var.inl>