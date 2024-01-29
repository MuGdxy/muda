#pragma once
#include <muda/ext/geo/lbvh/bvh.h>
#include <muda/ext/geo/lbvh/predicator.h>

namespace muda::lbvh
{
template <typename Real, typename Objects, typename AABBGetter, typename MortonCodeCalculator, typename OutputBackInserter>
MUDA_HOST uint32_t query(const BVH<Real, Objects, AABBGetter, MortonCodeCalculator>& tree,
                         const query_overlap<Real> q,
                         OutputBackInserter        outiter) noexcept
{
    using bvh_type   = BVH<Real, Objects, AABBGetter, MortonCodeCalculator>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    std::vector<std::size_t> stack;
    stack.reserve(64);
    stack.push_back(0);

    uint32_t num_found = 0;
    do
    {
        const index_type node = stack.back();
        stack.pop_back();
        const index_type L_idx = tree.host_nodes()[node].left_idx;
        const index_type R_idx = tree.host_nodes()[node].right_idx;

        if(intersects(q.target, tree.host_aabbs()[L_idx]))
        {
            const auto obj_idx = tree.host_nodes()[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                *outiter++ = obj_idx;
                ++num_found;
            }
            else  // the node is not a leaf.
            {
                stack.push_back(L_idx);
            }
        }
        if(intersects(q.target, tree.host_aabbs()[R_idx]))
        {
            const auto obj_idx = tree.host_nodes()[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                *outiter++ = obj_idx;
                ++num_found;
            }
            else  // the node is not a leaf.
            {
                stack.push_back(R_idx);
            }
        }
    } while(!stack.empty());
    return num_found;
}

template <typename Real, typename Objects, typename AABBGetter, typename MortonCodeCalculator, typename DistanceCalculator>
MUDA_HOST std::pair<uint32_t, Real> query(const BVH<Real, Objects, AABBGetter, MortonCodeCalculator>& tree,
                                          const query_nearest<Real>& q,
                                          DistanceCalculator calc_dist) noexcept
{
    using bvh_type   = BVH<Real, Objects, AABBGetter, MortonCodeCalculator>;
    using real_type  = typename bvh_type::real_type;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    //if(!tree.query_host_enabled())
    //{
    //    throw std::runtime_error("lbvh::bvh query_host is not enabled");
    //}

    // pair of {node_idx, mindist}
    std::vector<std::pair<index_type, real_type>> stack = {
        {0, mindist(tree.host_aabbs()[0], q.target)}};
    stack.reserve(64);

    uint32_t  nearest              = 0xFFFFFFFF;
    real_type current_nearest_dist = infinity<real_type>();
    do
    {
        const auto node = stack.back();
        stack.pop_back();
        if(node.second > current_nearest_dist)
        {
            // if aabb mindist > already_found_mindist, it cannot have a nearest
            continue;
        }

        const index_type L_idx = tree.host_nodes()[node.first].left_idx;
        const index_type R_idx = tree.host_nodes()[node.first].right_idx;

        const aabb_type& L_box = tree.host_aabbs()[L_idx];
        const aabb_type& R_box = tree.host_aabbs()[R_idx];

        const real_type L_mindist = mindist(L_box, q.target);
        const real_type R_mindist = mindist(R_box, q.target);

        const real_type L_minmaxdist = minmaxdist(L_box, q.target);
        const real_type R_minmaxdist = minmaxdist(R_box, q.target);

        // there should be an object that locates within minmaxdist.

        if(L_mindist <= R_minmaxdist)  // L is worth considering
        {
            const auto obj_idx = tree.host_nodes()[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)  // leaf node
            {
                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                if(dist <= current_nearest_dist)
                {
                    current_nearest_dist = dist;
                    nearest              = obj_idx;
                }
            }
            else
            {
                stack.emplace_back(L_idx, L_mindist);
            }
        }
        if(R_mindist <= L_minmaxdist)  // R is worth considering
        {
            const auto obj_idx = tree.host_nodes()[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)  // leaf node
            {
                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                if(dist <= current_nearest_dist)
                {
                    current_nearest_dist = dist;
                    nearest              = obj_idx;
                }
            }
            else
            {
                stack.emplace_back(R_idx, R_mindist);
            }
        }
    } while(!stack.empty());
    return std::make_pair(nearest, current_nearest_dist);
}
}  // namespace muda::lbvh
