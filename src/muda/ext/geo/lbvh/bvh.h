#pragma once
#include <muda/ext/geo/lbvh/bvh_viewer.h>

#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

namespace muda::lbvh
{
namespace details
{
    template <typename DerivedPolicy, typename UInt>
    void construct_internal_nodes(const thrust::detail::execution_policy_base<DerivedPolicy>& policy,
                                  Node*          nodes,
                                  UInt const*    node_code,
                                  const uint32_t num_objects)
    {
        thrust::for_each(
            policy,
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(num_objects - 1),
            [nodes, node_code, num_objects] __device__(const uint32_t idx)
            {
                nodes[idx].object_idx = 0xFFFFFFFF;  //  internal nodes

                const uint2 ij = determine_range(node_code, num_objects, idx);
                const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

                nodes[idx].left_idx  = gamma;
                nodes[idx].right_idx = gamma + 1;
                if(thrust::min(ij.x, ij.y) == gamma)
                {
                    nodes[idx].left_idx += num_objects - 1;
                }
                if(thrust::max(ij.x, ij.y) == gamma + 1)
                {
                    nodes[idx].right_idx += num_objects - 1;
                }
                nodes[nodes[idx].left_idx].parent_idx  = idx;
                nodes[nodes[idx].right_idx].parent_idx = idx;
                return;
            });
    }
}  // namespace details

template <typename Real, typename Object>
struct DefaultMortonCodeCalculator
{
    DefaultMortonCodeCalculator(AABB<Real> w)
        : whole(w)
    {
    }
    DefaultMortonCodeCalculator()                                   = default;
    ~DefaultMortonCodeCalculator()                                  = default;
    DefaultMortonCodeCalculator(DefaultMortonCodeCalculator const&) = default;
    DefaultMortonCodeCalculator(DefaultMortonCodeCalculator&&)      = default;
    DefaultMortonCodeCalculator& operator=(DefaultMortonCodeCalculator const&) = default;
    DefaultMortonCodeCalculator& operator=(DefaultMortonCodeCalculator&&) = default;

    __device__ __host__ inline uint32_t operator()(const Object&, const AABB<Real>& box) noexcept
    {
        auto p = centroid(box);
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        return morton_code(p);
    }
    AABB<Real> whole;
};

template <typename Real, typename Object, typename AABBGetter, typename MortonCodeCalculator = DefaultMortonCodeCalculator<Real, Object>>
class BVH
{
  public:
    using real_type                   = Real;
    using index_type                  = std::uint32_t;
    using object_type                 = Object;
    using aabb_type                   = AABB<real_type>;
    using node_type                   = details::Node;
    using aabb_getter_type            = AABBGetter;
    using morton_code_calculator_type = MortonCodeCalculator;

  public:
    BVH()                      = default;
    ~BVH()                     = default;
    BVH(const BVH&)            = default;
    BVH(BVH&&)                 = default;
    BVH& operator=(const BVH&) = default;
    BVH& operator=(BVH&&)      = default;

    void clear()
    {
        this->m_objects.clear();
        this->m_aabbs.clear();
        this->m_nodes.clear();
        return;
    }

    BVHViewer<real_type, object_type> viewer() noexcept
    {
        return BVHViewer<real_type, object_type>{
            static_cast<uint32_t>(m_nodes.size()),
            static_cast<uint32_t>(m_objects.size()),
            thrust::raw_pointer_cast(m_nodes.data()),
            thrust::raw_pointer_cast(m_aabbs.data()),
            thrust::raw_pointer_cast(m_objects.data())};
    }

    CBVHViewer<real_type, object_type> cviewer() const noexcept
    {
        return CBVHViewer<real_type, object_type>{
            static_cast<uint32_t>(m_nodes.size()),
            static_cast<uint32_t>(m_objects.size()),
            thrust::raw_pointer_cast(m_nodes.data()),
            thrust::raw_pointer_cast(m_aabbs.data()),
            thrust::raw_pointer_cast(m_objects.data())};
    }


    void build(cudaStream_t stream = nullptr)
    {
        auto policy = thrust::system::cuda::par_nosync.on(stream);
        //auto policy = thrust::device;

        if(m_objects.size() == 0u)
        {
            return;
        }

        m_host_dirty = true;

        const uint32_t num_objects        = m_objects.size();
        const uint32_t num_internal_nodes = num_objects - 1;
        const uint32_t num_nodes          = num_objects * 2 - 1;

        // --------------------------------------------------------------------
        // calculate morton code of each points

        const auto inf = std::numeric_limits<real_type>::infinity();
        aabb_type  default_aabb;
        default_aabb.upper.x = -inf;
        default_aabb.lower.x = inf;
        default_aabb.upper.y = -inf;
        default_aabb.lower.y = inf;
        default_aabb.upper.z = -inf;
        default_aabb.lower.z = inf;

        this->m_aabbs.resize(num_nodes, default_aabb);
        m_morton.resize(num_objects);
        m_indices.resize(num_objects);
        m_morton64.resize(num_objects);
        node_type default_node;
        default_node.parent_idx = 0xFFFFFFFF;
        default_node.left_idx   = 0xFFFFFFFF;
        default_node.right_idx  = 0xFFFFFFFF;
        default_node.object_idx = 0xFFFFFFFF;
        m_nodes.resize(num_nodes, default_node);
        m_flag_container.clear();
        m_flag_container.resize(num_internal_nodes, 0);

        thrust::transform(policy,
                          this->m_objects.begin(),
                          this->m_objects.end(),
                          m_aabbs.begin() + num_internal_nodes,
                          aabb_getter_type());

        const auto aabb_whole =
            thrust::reduce(policy,
                           m_aabbs.begin() + num_internal_nodes,
                           m_aabbs.end(),
                           default_aabb,
                           [] __device__ __host__(const aabb_type& lhs, const aabb_type& rhs)
                           { return merge(lhs, rhs); });


        thrust::transform(policy,
                          this->m_objects.begin(),
                          this->m_objects.end(),
                          m_aabbs.begin() + num_internal_nodes,
                          m_morton.begin(),
                          morton_code_calculator_type(aabb_whole));

        // --------------------------------------------------------------------
        // sort object-indices by morton code

        // iota the indices
        thrust::copy(policy,
                     thrust::make_counting_iterator<index_type>(0),
                     thrust::make_counting_iterator<index_type>(num_objects),
                     m_indices.begin());

        // keep indices ascending order
        thrust::stable_sort_by_key(
            policy,
            m_morton.begin(),
            m_morton.end(),
            thrust::make_zip_iterator(thrust::make_tuple(m_aabbs.begin() + num_internal_nodes,
                                                         m_indices.begin())));

        // --------------------------------------------------------------------
        // check morton codes are unique


        const auto uniqued = thrust::unique_copy(
            policy, m_morton.begin(), m_morton.end(), m_morton64.begin());

        const bool morton_code_is_unique = (m_morton64.end() == uniqued);
        if(!morton_code_is_unique)
        {
            thrust::transform(policy,
                              m_morton.begin(),
                              m_morton.end(),
                              m_indices.begin(),
                              m_morton64.begin(),
                              [] __device__ __host__(const uint32_t m, const uint32_t idx)
                              {
                                  unsigned long long int m64 = m;
                                  m64 <<= 32;
                                  m64 |= idx;
                                  return m64;
                              });
        }

        // --------------------------------------------------------------------
        // construct leaf nodes and aabbs

        thrust::transform(policy,
                          m_indices.begin(),
                          m_indices.end(),
                          this->m_nodes.begin() + num_internal_nodes,
                          [] __device__ __host__(const index_type idx)
                          {
                              node_type n;
                              n.parent_idx = 0xFFFFFFFF;
                              n.left_idx   = 0xFFFFFFFF;
                              n.right_idx  = 0xFFFFFFFF;
                              n.object_idx = idx;
                              return n;
                          });

        // --------------------------------------------------------------------
        // construct internal nodes

        if(morton_code_is_unique)
        {
            const uint32_t* node_code = thrust::raw_pointer_cast(m_morton.data());
            details::construct_internal_nodes(
                policy, thrust::raw_pointer_cast(m_nodes.data()), node_code, num_objects);
        }
        else  // 64bit version
        {
            const unsigned long long int* node_code =
                thrust::raw_pointer_cast(m_morton64.data());
            details::construct_internal_nodes(
                policy, thrust::raw_pointer_cast(m_nodes.data()), node_code, num_objects);
        }

        // --------------------------------------------------------------------
        // create AABB for each node by bottom-up strategy

        const auto flags = thrust::raw_pointer_cast(m_flag_container.data());

        thrust::for_each(policy,
                         thrust::make_counting_iterator<index_type>(num_internal_nodes),
                         thrust::make_counting_iterator<index_type>(num_nodes),
                         [nodes = thrust::raw_pointer_cast(m_nodes.data()),
                          aabbs = thrust::raw_pointer_cast(m_aabbs.data()),
                          flags] __device__(index_type idx)
                         {
                             uint32_t parent = nodes[idx].parent_idx;
                             while(parent != 0xFFFFFFFF)  // means idx == 0
                             {
                                 const int old = atomicCAS(flags + parent, 0, 1);
                                 if(old == 0)
                                 {
                                     // this is the first thread entered here.
                                     // wait the other thread from the other child node.
                                     return;
                                 }
                                 MUDA_KERNEL_ASSERT(old == 1,"old=%d",old);
                                 // here, the flag has already been 1. it means that this
                                 // thread is the 2nd thread. merge AABB of both childlen.

                                 const auto lidx = nodes[parent].left_idx;
                                 const auto ridx = nodes[parent].right_idx;
                                 const auto lbox = aabbs[lidx];
                                 const auto rbox = aabbs[ridx];
                                 aabbs[parent]   = merge(lbox, rbox);

                                 // look the next parent...
                                 parent = nodes[parent].parent_idx;
                             }
                             return;
                         });
    }

    const auto& objects() const noexcept { return m_objects; }
    auto&       objects() noexcept { return m_objects; }
    const auto& aabbs() const noexcept { return m_aabbs; }
    const auto& nodes() const noexcept { return m_nodes; }

    const auto& host_objects() const noexcept
    {
        download_if_dirty();
        return m_h_objects;
    }
    const auto& host_aabbs() const noexcept
    {
        download_if_dirty();
        return m_h_aabbs;
    }
    const auto& host_nodes() const noexcept
    {
        download_if_dirty();
        return m_h_nodes;
    }

  private:
    muda::DeviceVector<uint32_t>               m_morton;
    muda::DeviceVector<uint32_t>               m_indices;
    muda::DeviceVector<unsigned long long int> m_morton64;
    muda::DeviceVector<int>                    m_flag_container;

    muda::DeviceVector<object_type> m_objects;
    muda::DeviceVector<aabb_type>   m_aabbs;
    muda::DeviceVector<node_type>   m_nodes;

    mutable bool                             m_host_dirty = true;
    mutable thrust::host_vector<object_type> m_h_objects;
    mutable thrust::host_vector<aabb_type>   m_h_aabbs;
    mutable thrust::host_vector<node_type>   m_h_nodes;

    void download_if_dirty() const
    {
        if(m_host_dirty)
        {
            m_h_objects  = m_objects;
            m_h_aabbs    = m_aabbs;
            m_h_nodes    = m_nodes;
            m_host_dirty = false;
        }
    }
};
}  // namespace muda::lbvh
