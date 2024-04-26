#pragma once
#include <muda/viewer/viewer_base.h>
#include <muda/container/vector.h>
#include <muda/ext/geo/lbvh/aabb.h>
#include <muda/ext/geo/lbvh/morton_code.h>
#include <muda/ext/geo/lbvh/predicator.h>

namespace muda::lbvh
{
template <typename Real, typename Object, typename AABBGetter, typename MortonCodeCalculator>
class BVH;

namespace details
{
    struct Node
    {
        std::uint32_t parent_idx;  // parent node
        std::uint32_t left_idx;    // index of left  child node
        std::uint32_t right_idx;   // index of right child node
        std::uint32_t object_idx;  // == 0xFFFFFFFF if internal node.
    };

    template <typename UInt>
    MUDA_GENERIC uint2 determine_range(UInt const* node_code, const uint32_t num_leaves, uint32_t idx)
    {
        if(idx == 0)
        {
            return make_uint2(0, num_leaves - 1);
        }

        // determine direction of the range
        const UInt self_code = node_code[idx];
        const int  L_delta   = common_upper_bits(self_code, node_code[idx - 1]);
        const int  R_delta   = common_upper_bits(self_code, node_code[idx + 1]);
        const int  d         = (R_delta > L_delta) ? 1 : -1;

        // Compute upper bound for the length of the range

        const int delta_min = thrust::min(L_delta, R_delta);
        int       l_max     = 2;
        int       delta     = -1;
        int       i_tmp     = idx + d * l_max;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        while(delta > delta_min)
        {
            l_max <<= 1;
            i_tmp = idx + d * l_max;
            delta = -1;
            if(0 <= i_tmp && i_tmp < num_leaves)
            {
                delta = common_upper_bits(self_code, node_code[i_tmp]);
            }
        }

        // Find the other end by binary search
        int l = 0;
        int t = l_max >> 1;
        while(t > 0)
        {
            i_tmp = idx + (l + t) * d;
            delta = -1;
            if(0 <= i_tmp && i_tmp < num_leaves)
            {
                delta = common_upper_bits(self_code, node_code[i_tmp]);
            }
            if(delta > delta_min)
            {
                l += t;
            }
            t >>= 1;
        }
        uint32_t jdx = idx + l * d;
        if(d < 0)
        {
            thrust::swap(idx, jdx);  // make it sure that idx < jdx
        }
        return make_uint2(idx, jdx);
    }

    template <typename UInt>
    MUDA_GENERIC uint32_t find_split(UInt const*    node_code,
                                     const uint32_t num_leaves,
                                     const uint32_t first,
                                     const uint32_t last) noexcept
    {
        const UInt first_code = node_code[first];
        const UInt last_code  = node_code[last];
        if(first_code == last_code)
        {
            return (first + last) >> 1;
        }
        const int delta_node = common_upper_bits(first_code, last_code);

        // binary search...
        int split  = first;
        int stride = last - first;
        do
        {
            stride           = (stride + 1) >> 1;
            const int middle = split + stride;
            if(middle < last)
            {
                const int delta = common_upper_bits(first_code, node_code[middle]);
                if(delta > delta_node)
                {
                    split = middle;
                }
            }
        } while(stride > 1);

        return split;
    }


    template <bool IsConst, typename Real, typename Object>
    class BVHViewerBase : muda::ViewerBase<IsConst>
    {
        MUDA_VIEWER_COMMON_NAME(BVHViewerBase);

        using Base = muda::ViewerBase<IsConst>;
        template <typename U>
        using auto_const_t = typename Base::template auto_const_t<U>;

        template <typename Real_, typename Object_, typename AABBGetter, typename MortonCodeCalculator>
        friend class BVH;

      public:
        using real_type   = Real;
        using aabb_type   = AABB<real_type>;
        using node_type   = details::Node;
        using index_type  = std::uint32_t;
        using object_type = Object;


        using ConstViewer    = BVHViewerBase<true, real_type, object_type>;
        using NonConstViewer = BVHViewerBase<false, real_type, object_type>;
        using ThisViewer     = BVHViewerBase<IsConst, real_type, object_type>;

        struct DefaultQueryCallback
        {
            MUDA_GENERIC void operator()(uint32_t obj_idx) const noexcept {}
        };

        MUDA_GENERIC BVHViewerBase(const uint32_t             num_nodes,
                                   const uint32_t             num_objects,
                                   auto_const_t<node_type>*   nodes,
                                   auto_const_t<aabb_type>*   aabbs,
                                   auto_const_t<object_type>* objects)
            : m_num_nodes(num_nodes)
            , m_num_objects(num_objects)
            , m_nodes(nodes)
            , m_aabbs(aabbs)
            , m_objects(objects)
        {
            MUDA_KERNEL_ASSERT(m_nodes && m_aabbs && m_objects,
                               "BVHViewerBase[%s:%s]: nullptr is passed,"
                               "nodes=%p,"
                               "aabbs=%p,"
                               "objects=%p\n",
                               this->name(),
                               this->kernel_name(),
                               m_nodes,
                               m_aabbs,
                               m_objects);
        }

        MUDA_GENERIC auto as_const() const noexcept
        {
            return ConstViewer{m_num_nodes, m_num_objects, m_nodes, m_aabbs, m_objects};
        }

        MUDA_GENERIC operator ConstViewer() const noexcept
        {
            return as_const();
        }

        MUDA_GENERIC auto num_nodes() const noexcept { return m_num_nodes; }
        MUDA_GENERIC auto num_objects() const noexcept { return m_num_objects; }

        // query object indices that potentially overlaps with query aabb.
        //
        // requirements:
        //  - F: void (uin32_t obj_idx)
        template <typename F, uint32_t StackNum = 64>
        MUDA_GENERIC uint32_t query(const query_overlap<real_type>& q,
                                    F callback = DefaultQueryCallback{}) const noexcept
        {
            index_type  stack[StackNum];
            index_type* stack_ptr = stack;
            index_type* stack_end = stack + StackNum;
            *stack_ptr++          = 0;  // root node is always 0

            uint32_t num_found = 0;
            do
            {
                const index_type node  = *--stack_ptr;
                const index_type L_idx = m_nodes[node].left_idx;
                const index_type R_idx = m_nodes[node].right_idx;

                if(intersects(q.target, m_aabbs[L_idx]))
                {
                    const auto obj_idx = m_nodes[L_idx].object_idx;
                    if(obj_idx != 0xFFFFFFFF)
                    {
                        if constexpr(!std::is_same_v<F, DefaultQueryCallback>)
                        {
                            callback(obj_idx);
                        }
                        ++num_found;
                    }
                    else  // the node is not a leaf.
                    {
                        *stack_ptr++ = L_idx;
                    }
                }
                if(intersects(q.target, m_aabbs[R_idx]))
                {
                    const auto obj_idx = m_nodes[R_idx].object_idx;
                    if(obj_idx != 0xFFFFFFFF)
                    {
                        if constexpr(!std::is_same_v<F, DefaultQueryCallback>)
                        {
                            callback(obj_idx);
                        }
                        ++num_found;
                    }
                    else  // the node is not a leaf.
                    {
                        *stack_ptr++ = R_idx;
                    }
                }
                MUDA_KERNEL_ASSERT(stack_ptr < stack_end,
                                   "LBVHQuery[%s:%s]: stack overflow, try use a larger StackNum.",
                                   this->name(),
                                   this->kernel_name());
            } while(stack < stack_ptr);
            return num_found;
        }

        // query object index that is the nearst to the query point.
        //
        // requirements:
        // - FDistanceCalculator must be able to calc distance between a point to an object.
        //   which means FDistanceCalculator: Real (const object_type& lhs, const object_type& rhs)
        //
        template <typename FDistanceCalculator, uint32_t StackNum = 64>
        MUDA_GENERIC thrust::pair<uint32_t, real_type> query(
            const query_nearest<real_type>& q, FDistanceCalculator calc_dist) const noexcept
        {
            // pair of {node_idx, mindist}
            thrust::pair<index_type, real_type>  stack[StackNum];
            thrust::pair<index_type, real_type>* stack_ptr = stack;
            thrust::pair<index_type, real_type>* stack_end = stack + StackNum;

            *stack_ptr++ = thrust::make_pair(0, mindist(m_aabbs[0], q.target));

            uint32_t  nearest                = 0xFFFFFFFF;
            real_type dist_to_nearest_object = infinity<real_type>();
            do
            {
                const auto node = *--stack_ptr;
                if(node.second > dist_to_nearest_object)
                {
                    // if aabb mindist > already_found_mindist, it cannot have a nearest
                    continue;
                }

                const index_type L_idx = m_nodes[node.first].left_idx;
                const index_type R_idx = m_nodes[node.first].right_idx;

                const aabb_type& L_box = m_aabbs[L_idx];
                const aabb_type& R_box = m_aabbs[R_idx];

                const real_type L_mindist = mindist(L_box, q.target);
                const real_type R_mindist = mindist(R_box, q.target);

                const real_type L_minmaxdist = minmaxdist(L_box, q.target);
                const real_type R_minmaxdist = minmaxdist(R_box, q.target);

                // there should be an object that locates within minmaxdist.

                if(L_mindist <= R_minmaxdist)  // L is worth considering
                {
                    const auto obj_idx = m_nodes[L_idx].object_idx;
                    if(obj_idx != 0xFFFFFFFF)  // leaf node
                    {
                        const real_type dist = calc_dist(q.target, m_objects[obj_idx]);
                        if(dist <= dist_to_nearest_object)
                        {
                            dist_to_nearest_object = dist;
                            nearest                = obj_idx;
                        }
                    }
                    else
                    {
                        *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
                    }
                }
                if(R_mindist <= L_minmaxdist)  // R is worth considering
                {
                    const auto obj_idx = m_nodes[R_idx].object_idx;
                    if(obj_idx != 0xFFFFFFFF)  // leaf node
                    {
                        const real_type dist = calc_dist(q.target, m_objects[obj_idx]);
                        if(dist <= dist_to_nearest_object)
                        {
                            dist_to_nearest_object = dist;
                            nearest                = obj_idx;
                        }
                    }
                    else
                    {
                        *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
                    }
                }
                MUDA_KERNEL_ASSERT(stack_ptr < stack_end,
                                   "LBVHQuery[%s:%s]: stack overflow, try use a larger StackNum.",
                                   this->name(),
                                   this->kernel_name());
            } while(stack < stack_ptr);
            return thrust::make_pair(nearest, dist_to_nearest_object);
        }

        MUDA_GENERIC auto_const_t<object_type>& object(const uint32_t idx) noexcept
        {
            check_index(idx);
            return m_objects[idx];
        }

        MUDA_GENERIC const object_type& object(const uint32_t idx) const noexcept
        {
            check_index(idx);
            return m_objects[idx];
        }

      private:
        uint32_t m_num_nodes;  // (# of internal node) + (# of leaves), 2N+1
        uint32_t m_num_objects;  // (# of leaves), the same as the number of objects

        auto_const_t<node_type>*   m_nodes;
        auto_const_t<aabb_type>*   m_aabbs;
        auto_const_t<object_type>* m_objects;

        MUDA_INLINE MUDA_GENERIC void check_index(const uint32_t idx) const noexcept
        {
            MUDA_KERNEL_ASSERT(idx < m_num_objects,
                               "BVHViewer[%s:%s]: index out of range, idx=%u, num_objects=%u",
                               this->name(),
                               this->kernel_name(),
                               idx,
                               m_num_objects);
        }
    };
}  // namespace details

template <typename Real, typename Object>
using BVHViewer = details::BVHViewerBase<false, Real, Object>;
template <typename Real, typename Object>
using CBVHViewer = details::BVHViewerBase<true, Real, Object>;
}  // namespace muda::lbvh

namespace muda
{
template <typename Real, typename Object>
struct read_only_viewer<lbvh::BVHViewer<Real, Object>>
{
    using type = lbvh::CBVHViewer<Real, Object>;
};

template <typename Real, typename Object>
struct read_write_viewer<lbvh::CBVHViewer<Real, Object>>
{
    using type = lbvh::BVHViewer<Real, Object>;
};
}  // namespace muda