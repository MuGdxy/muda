#include <muda/compute_graph/compute_graph_var_manager.h>

namespace muda
{
MUDA_INLINE void ComputeGraphNodeBase::graphviz_id(std::ostream& o,
                                                   const ComputeGraphGraphvizOptions& options) const
{
    o << "node_g" << options.graph_id << "_n" << node_id();
}

MUDA_INLINE void ComputeGraphNodeBase::graphviz_def(std::ostream& o,
                                                    const ComputeGraphGraphvizOptions& options) const
{
    graphviz_id(o, options);
    o << "[";
    if(!name().empty())
        o << "label=\"" << name() << "\", ";
    o << options.node_style;
    
    o << "]";
}

MUDA_INLINE void ComputeGraphNodeBase::graphviz_var_usages(std::ostream& o,
                                                           const ComputeGraphGraphvizOptions& options) const
{
    for(auto&& [var_id, usage] : var_usages())
    {
        auto var = m_graph->m_var_manager->m_vars[var_id.value()];
        var->graphviz_id(o, options);
        o << "->";
        graphviz_id(o, options);
        if(usage == ComputeGraphVarUsage::ReadWrite)
            o << "[" << options.read_write_style << "]";
        else
            o << "[" << options.read_style << "]";
        o << "\n";
    }
}

MUDA_INLINE void ComputeGraphNodeBase::set_deps_range(size_t begin, size_t count)
{
    m_deps_begin = begin;
    m_deps_count = count;
}

template <typename NodeT, ComputeGraphNodeType Type>
MUDA_INLINE void ComputeGraphNode<NodeT, Type>::set_node(S<NodeT> node)
{
    m_node = node;
    set_handle(m_node->handle());
}
}  // namespace muda