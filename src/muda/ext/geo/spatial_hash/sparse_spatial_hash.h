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
    using Vector3i = Eigen::Vector3i;
    using I32      = int32_t;

    struct
    {
        I32 pass : 3;
        I32 home : 3;
        I32 overlap : 8;
    } ctlbit;  // controll bit

    I32      cid;  // cell id
    I32      oid;
    Vector3i ijk;

    MUDA_GENERIC SpatialPartitionCell()
        : cid(~0)
        , oid(~0)
        , ijk(-Vector3i::Ones())
    {
        ctlbit.home    = 0;
        ctlbit.overlap = 0;
        ctlbit.pass    = 0;
    }

    MUDA_GENERIC SpatialPartitionCell(I32 cid, I32 oid)
        : cid(cid)
        , oid(oid)
        , ijk(-Vector3i::Ones())
    {
        ctlbit.home    = 0;
        ctlbit.overlap = 0;
    }

    MUDA_GENERIC bool is_phantom() const { return ctlbit.home != ctlbit.pass; }

    MUDA_GENERIC bool is_home() const { return ctlbit.home == ctlbit.pass; }

    MUDA_GENERIC void set_as_phantom(const Vector3i& home_ijk, const Vector3i& cell_ijk)
    {
        ctlbit.pass = pass_type(cell_ijk);
        ctlbit.home = pass_type(home_ijk);
    }

    MUDA_GENERIC void set_as_home(const Vector3i& ijk)
    {
        // bit   2           1           0
        // home  (i % 2)     (j % 2)     (k % 2)
        ctlbit.home = pass_type(ijk);
        ctlbit.pass = ctlbit.home;
        ctlbit.overlap |= (1 << ctlbit.home);
    }

    MUDA_GENERIC void set_overlap(const Vector3i& ijk)
    {
        ctlbit.overlap |= (1 << pass_type(ijk));
    }

    MUDA_GENERIC static I32 pass_type(const Vector3i& ijk)
    {
        return (((I32)ijk(0) % 2) << 2) | (((I32)ijk(1) % 2) << 1)
               | (((I32)ijk(2) % 2) << 0);
    }

    MUDA_GENERIC static bool allow_ignore(const SpatialPartitionCell& l,
                                          const SpatialPartitionCell& r)
    {
//        if(l.is_phantom() && r.is_phantom())
//        {
//            return true;
//        }
//
//        const SpatialPartitionCell* arr[] = {&l, &r};
//
//        I32 pass           = l.ctlbit.pass;
//        I32 common_overlap = l.ctlbit.overlap & r.ctlbit.overlap;
//#pragma unroll
//        for(I32 i = 0; i < 2; ++i)
//        {
//            I32 encode_home = (1 << arr[i]->ctlbit.home);
//            if(arr[i]->ctlbit.home < pass && (common_overlap & encode_home))
//            {
//                return true;
//            }
//        }
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

template <typename Hash = Morton<int32_t>>
class SpatialHashTableInfo
{
    using Float    = float;
    using I32      = int32_t;
    using Vector3  = Eigen::Vector<Float, 3>;
    using Vector3i = Eigen::Vector<int, 3>;

  public:
    Float   cell_size = 0.0f;
    Vector3 coord_min = Vector3::Zero();

    MUDA_GENERIC SpatialHashTableInfo() = default;

    MUDA_GENERIC SpatialHashTableInfo(Float cell_size, const Vector3& coord_min)
        : cell_size(cell_size)
        , coord_min(coord_min)
    {
    }

    MUDA_GENERIC I32 hash_cell(const Vector3& xyz) const
    {
        return hash_cell(cell(xyz));
    }

    MUDA_GENERIC I32 hash_cell(const Vector3i& ijk) const
    {
        return Hash()(ijk) % 0x40000000;
    }

    MUDA_GENERIC Vector3i cell(const Vector3& xyz) const
    {
        Vector3i ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = (xyz(i) - coord_min(i)) / cell_size;
        return ret;
    }
    MUDA_GENERIC Vector3 coord(const Vector3i& ijk) const
    {
        Vector3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = ijk(i) * cell_size + coord_min(i);
        return ret;
    }

    MUDA_GENERIC Vector3 cell_center_coord(const Vector3i& ijk) const
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

template <typename Hash>
class SpatialPartitionLauncher;

namespace details
{
    template <typename Hash = Morton<int32_t>>
    class SparseSpatialHashImpl
    {
      public:
        using Cell    = SpatialPartitionCell;
        using Vector3 = Eigen::Vector3f;

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
        void beginCreateCollisionPairs(CBufferView<BoundingSphere> boundingSphereList,
                                       DeviceBuffer<CollisionPair>& collisionPairs,
                                       Pred&& pred);

        DeviceBuffer<float>   allRadius;
        DeviceBuffer<Vector3> allCoords;

        void beginCalculateCellSizeAndCoordMin();

        void beginSetupHashTable();

        void beginFillHashCells();

        void beginSortHashCells();

        void beginCountObjectPerCell();

        void waitAndCreateTempData();

        template <typename Pred>
        void beginCountCollisionPairs(Pred&& pred);

        void waitAndAllocCollisionPairList(DeviceBuffer<CollisionPair>& collisionPairs);

        template <typename Pred>
        void beginSetupCollisionPairList(DeviceBuffer<CollisionPair>& collisionPairs,
                                         Pred&& pred);
    };
}  // namespace details

using Hash = Morton<int32_t>;
//template <typename Hash = Morton<int32_t>>
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
        m_impl.beginCreateCollisionPairs(spheres, collisionPairs, std::forward<Pred>(pred));
    }
};
}  // namespace muda::spatial_hash

#include "details/sparse_spatial_hash.inl"
