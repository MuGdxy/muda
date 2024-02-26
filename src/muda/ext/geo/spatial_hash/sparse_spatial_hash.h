#pragma once
#include <muda/muda_def.h>
#include <muda/ext/geo/spatial_hash/morton_hash.h>
#include <muda/launch/launch.h>
#include <muda/launch/parallel_for.h>
#include <muda/buffer/device_buffer.h>

#include <muda/cub/device/device_reduce.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/ext/geo/spatial_hash/bounding_volume.h>

namespace muda::spatial_hash
{
/**
 * \brief To represent a cell-object pair in the spatial hash 3D grid 
 * e.g. (cell_id,object_id) = (1024, 32) for the meaning: the 32th object overlap with the 1024th cell
 */
class SpatialPartitionCell
{
  public:
    using Vector3u = Eigen::Vector3<uint32_t>;
    using U32      = uint32_t;

    struct
    {
        // most use unsigned int to avoid comparison problem
        U32 pass : 3;
        U32 home : 3;
        U32 overlap : 8;
    } ctlbit;  // controll bit

    U32 cid;  // cell id
    U32 oid;
    // Vector3u ijk;

    MUDA_GENERIC SpatialPartitionCell()
        : cid(~0u)
        , oid(~0u)
    //, ijk(Vector3u::Zero())
    {
        ctlbit.home    = 0u;
        ctlbit.overlap = 0u;
        ctlbit.pass    = 0u;
    }

    MUDA_GENERIC SpatialPartitionCell(U32 cid, U32 oid)
        : cid(cid)
        , oid(oid)
    //, ijk(Vector3u::Zero())
    {
        ctlbit.home    = 0u;
        ctlbit.overlap = 0u;
    }

    MUDA_GENERIC bool is_phantom() const { return ctlbit.home != ctlbit.pass; }

    MUDA_GENERIC bool is_home() const { return ctlbit.home == ctlbit.pass; }

    MUDA_GENERIC void set_as_phantom(const Vector3u& home_ijk, const Vector3u& cell_ijk)
    {
        ctlbit.pass = pass_type(cell_ijk);
        ctlbit.home = pass_type(home_ijk);
    }

    MUDA_GENERIC void set_as_home(const Vector3u& ijk)
    {
        // bit   2           1           0
        // home  (i % 2)     (j % 2)     (k % 2)
        ctlbit.home = pass_type(ijk);
        ctlbit.pass = ctlbit.home;
        ctlbit.overlap |= (1 << ctlbit.home);
    }

    MUDA_GENERIC void set_overlap(const Vector3u& ijk)
    {
        ctlbit.overlap |= (1 << pass_type(ijk));
    }

    MUDA_GENERIC static U32 pass_type(const Vector3u& ijk)
    {
        return (((U32)ijk(0) % 2) << 2) | (((U32)ijk(1) % 2) << 1)
               | (((U32)ijk(2) % 2) << 0);
    }

    MUDA_GENERIC static bool allow_ignore(const SpatialPartitionCell& l,
                                          const SpatialPartitionCell& r)
    {
        if(l.is_phantom() && r.is_phantom())
        {
            return true;
        }

        const SpatialPartitionCell* arr[] = {&l, &r};

        U32 pass           = l.ctlbit.pass;
        U32 common_overlap = l.ctlbit.overlap & r.ctlbit.overlap;
#pragma unroll
        for(U32 i = 0; i < 2; ++i)
        {
            U32 encode_home = (1 << arr[i]->ctlbit.home);
            if(arr[i]->ctlbit.home < pass && (common_overlap & encode_home))
            {
                return true;
            }
        }
        return false;
    }

    //static void csv_header(std::ostream& os)
    //{
    //    os << "cid,oid,pass,home,overlap,i,j,k" << std::endl;
    //}

    //friend std::ostream& operator<<(std::ostream& os, const SpatialPartitionCell& cell)
    //{
    //    os << std::hex << cell.cid << "," << std::dec << cell.oid << ","
    //       << std::hex << cell.ctlbit.pass << "," << cell.ctlbit.home << ","
    //       << cell.ctlbit.overlap << "," << std::dec << cell.ijk(0) << ","
    //       << cell.ijk(1) << "," << cell.ijk(2);
    //    return os;
    //}
};

template <typename Hash = Morton<uint32_t>>
class SpatialHashTableInfo
{
    using Float    = float;
    using Vector3  = Eigen::Vector<Float, 3>;
    using Vector3i = Eigen::Vector<int, 3>;
    using Vector3u = Eigen::Vector<uint32_t, 3>;
    using U32      = uint32_t;

  public:
    Float   cell_size = 0.0f;
    Vector3 coord_min = Vector3::Zero();

    MUDA_GENERIC SpatialHashTableInfo() = default;

    MUDA_GENERIC SpatialHashTableInfo(Float cell_size, const Vector3& coord_min)
        : cell_size(cell_size)
        , coord_min(coord_min)
    {
    }

    MUDA_GENERIC U32 hash_cell(const Vector3& xyz) const
    {
        return hash_cell(cell(xyz));
    }

    MUDA_GENERIC U32 hash_cell(const Vector3u& ijk) const
    {
        return Hash()(ijk) % 0x40000000;
    }

    MUDA_GENERIC Vector3u cell(const Vector3& xyz) const
    {
        Vector3u ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = (xyz(i) - coord_min(i)) / cell_size;
        return ret;
    }
    MUDA_GENERIC Vector3 coord(const Vector3u& ijk) const
    {
        Vector3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = ijk(i) * cell_size + coord_min(i);
        return ret;
    }

    MUDA_GENERIC Vector3 cell_center_coord(const Vector3u& ijk) const
    {
        Vector3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = (ijk(i) + 0.5f) * cell_size + coord_min(i);
        return ret;
    }
};

class CollisionPair
{
  public:
    Eigen::Vector2i id;

    MUDA_GENERIC CollisionPair() = default;

    MUDA_GENERIC CollisionPair(int i, int j)
        : id(i, j)
    {
    }

    MUDA_GENERIC friend bool operator<(const CollisionPair& l, const CollisionPair& r)
    {
        return (l.id[0] < r.id[0]) || (l.id[0] == r.id[0] && l.id[1] < r.id[1]);
    }

    MUDA_GENERIC friend bool operator==(const CollisionPair& l, const CollisionPair& r)
    {
        return (l.id[0] == r.id[0] && l.id[1] == r.id[1]);
    }

    friend std::ostream& operator<<(std::ostream& os, const CollisionPair& c)
    {
        os << "(" << c.id[0] << "," << c.id[1] << ")";
        return os;
    }
};

class DefaultPredication
{
  public:
    __device__ bool operator()(int i, int j) { return true; }
};

namespace details
{
    template <typename Hash = Morton<uint32_t>>
    class SparseSpatialHashImpl
    {
      public:
        using Cell     = SpatialPartitionCell;
        using U32      = uint32_t;
        using I32      = int32_t;
        using Vector3u = Eigen::Vector3<U32>;
        using Vector3i = Eigen::Vector3<I32>;
        using Vector3  = Eigen::Vector3f;

        muda::Stream& m_stream;

        CBufferView<BoundingSphere> spheres;

        DeviceVar<int>     cellCount;
        DeviceVar<int>     pairCount;
        DeviceVar<float>   maxRadius;
        DeviceVar<Vector3> minCoord;

        DeviceVar<SpatialHashTableInfo<Hash>> spatialHashConfig;
        SpatialHashTableInfo<Hash>            h_spatialHashConfig;

        DeviceBuffer<SpatialPartitionCell> cellArrayValue;
        DeviceBuffer<SpatialPartitionCell> cellArrayValueSorted;
        DeviceBuffer<int>                  cellArrayKey;
        DeviceBuffer<int>                  cellArrayKeySorted;

        DeviceBuffer<int> uniqueKey;
        DeviceVar<int>    uniqueKeyCount;
        int               validCellCount;
        int               sum;
        size_t            pairListOffset = 0;

        DeviceBuffer<int> objCountInCell;
        DeviceBuffer<int> objCountInCellPrefixSum;

        DeviceBuffer<int> collisionPairCount;
        DeviceBuffer<int> collisionPairPrefixSum;

        int level = 0;

        //using Hash = Hash;
        SparseSpatialHashImpl(muda::Stream& stream = muda::Stream::Default())
            : m_stream(stream)
        {
        }

        template <typename Pred>
        void detect(CBufferView<BoundingSphere>  boundingSphereList,
                    DeviceBuffer<CollisionPair>& collisionPairs,
                    Pred&&                       pred);

        DeviceBuffer<float>   allRadius;
        DeviceBuffer<Vector3> allCoords;
        DeviceBuffer<int>     collisionPairUpperBoundPerCell;
        DeviceBuffer<int>     collisionPairUpperBoundPerCellPrefixSum;

        void calculate_hash_table_basic_info();

        void setup_hash_table();

        void fill_hash_cells();

        void count_object_per_cell();

        template <typename Pred>
        void simple_setup_collision_pairs(Pred&& pred, DeviceBuffer<CollisionPair>& collisionPairs);

        template <typename Pred>
        void simple_count_collision_pairs(Pred&& pred);

        void alloc_collision_pair_list(DeviceBuffer<CollisionPair>& collisionPairs,
                                       int totalCollisionPairCount);

        template <typename Pred>
        void simple_fill_collision_pair_list(DeviceBuffer<CollisionPair>& collisionPairs,
                                             Pred&& pred);

        template <typename Pred>
        void blanced_count_collision_pairs(Pred&& pred);
    };
}  // namespace details

template <typename Hash = Morton<uint32_t>>
class SparseSpatialHash
{
    // algorithm comes from:
    // https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  private:
    using Impl = details::SparseSpatialHashImpl<Hash>;

    Impl m_impl;

  public:
    SparseSpatialHash(muda::Stream& stream = muda::Stream::Default())
        : m_impl(stream)
    {
    }

    template <typename Pred = DefaultPredication>
    void detect(CBufferView<BoundingSphere>  spheres,
                DeviceBuffer<CollisionPair>& collisionPairs,
                Pred&&                       pred = {})
    {
        m_impl.detect(spheres, collisionPairs, std::forward<Pred>(pred));
    }
};
}  // namespace muda::spatial_hash

#include "details/sparse_spatial_hash.inl"
