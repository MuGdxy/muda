#pragma once

namespace muda
{
MUDA_INLINE void ComputeGraphNodeBase::graphviz_id(std::ostream& o) const
{
    o << "node_" << node_id();
}
MUDA_INLINE void ComputeGraphNodeBase::graphviz_def(std::ostream& o) const
{
    graphviz_id(o);
    o << "[";
    if(!name().empty())
        o << "label=\"" << name() << "\", ";
    o << R"(shape="egg", color="#82B366", style="filled", fillcolor="#D5E8D4"])";
}
MUDA_INLINE void ComputeGraphNodeBase::graphviz_var_usages(std::ostream& o) const
{
    for(auto&& [var_id, usage] : var_usages())
    {
        auto var = m_graph->m_vars[var_id.value()];
        var->graphviz_id(o);
        o << "->";
        graphviz_id(o);
        if(usage == ComputeGraphVarUsage::ReadWrite)
            o << R"([color="#F08E81", arrowhead = diamond, ])";
        else
            o << R"([color="#64BBE2", arrowhead = dot, ])";
        o << "\n";
    }
}
}  // namespace muda