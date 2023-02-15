#pragma once
#include <muda/muda_def.h>
#include <muda/encode/morton.h>
#include <muda/viewer/dense.h>
#include <muda/launch/launch.h>

#include <muda/buffer/device_buffer.h>
#include <muda/container.h>
#include <muda/composite/cse.h>

#include <muda/algorithm/device_reduce.h>
#include <muda/algorithm/device_radix_sort.h>
#include <muda/algorithm/device_run_length_encode.h>
#include <muda/algorithm/device_scan.h>
#include <muda/encode/hash.h>
#include <muda/encode/morton.h>

#include "bounding_volume.h"
#include "collide.h"

namespace muda
{
/// <summary>
/// To represent a cell-object pair in the spatial hash 3D grid
/// e.g. (cell_id,object_id) = (1024, 32) for the meaning: the 32th object overlap with the 1024th cell
/// </summary>
class SpatialPartitionCell
{
  public:
    using ivec3 = Eigen::Vector3<int>;
    using u32   = uint32_t;

    struct
    {
        u32 pass : 3;
        u32 home : 3;
        u32 overlap : 8;
    } ctlbit;  // controll bit

    u32   cid;  // cell id
    u32   oid;
    ivec3 ijk;

    MUDA_GENERIC SpatialPartitionCell()
        : cid(-1)
        , oid(-1)
    {
        ctlbit.home    = 0;
        ctlbit.overlap = 0;
        ijk            = ivec3(-1, -1, -1);
    }

    MUDA_GENERIC SpatialPartitionCell(u32 cid, u32 oid)
        : cid(cid)
        , oid(oid)
    {
        ctlbit.home    = 0;
        ctlbit.overlap = 0;
        ijk            = ivec3(-1, -1, -1);
    }

    MUDA_GENERIC bool isPhantom() const { return ctlbit.home != ctlbit.pass; }

    MUDA_GENERIC bool isHome() const { return ctlbit.home == ctlbit.pass; }

    MUDA_GENERIC void setAsPhantom(const ivec3& home_ijk, const ivec3& cell_ijk)
    {
        ctlbit.pass = passType(cell_ijk);
        ctlbit.home = passType(home_ijk);
    }

    MUDA_GENERIC void setAsHome(const ivec3& ijk)
    {
        //bit		2			1			0
        //home		(i % 2)		(j % 2)		(k % 2)
        ctlbit.home = passType(ijk);
        ctlbit.pass = ctlbit.home;
        ctlbit.overlap |= (1 << ctlbit.home);
    }

    MUDA_GENERIC void setOverlap(const ivec3& ijk)
    {
        ctlbit.overlap |= (1 << passType(ijk));
    }

    MUDA_GENERIC static u32 passType(const ivec3& ijk)
    {
        return (((u32)ijk(0) % 2) << 2) | (((u32)ijk(1) % 2) << 1)
               | (((u32)ijk(2) % 2) << 0);
    }

    MUDA_GENERIC static bool allowIgnore(const SpatialPartitionCell& l,
                                         const SpatialPartitionCell& r)
    {
        if(l.isPhantom() && r.isPhantom())
        {
            //muda_kernel_printf("phantom, cid(%d,%d) home=(%d,%d) pass(%d,%d) eid(%d,%d) ijk{(%d,%d,%d), (%d,%d,%d)}\n",
            //                   l.cid,
            //                   r.cid,
            //                   l.ctlbit.home,
            //                   r.ctlbit.home,
            //                   l.ctlbit.pass,
            //                   r.ctlbit.pass,
            //                   l.oid,
            //                   r.oid,
            //                   l.ijk(0),
            //                   l.ijk(1),
            //                   l.ijk(2),
            //                   r.ijk(0),
            //                   r.ijk(1),
            //                   r.ijk(2));
            return true;
        }

        const SpatialPartitionCell* arr[] = {&l, &r};

        u32 pass           = l.ctlbit.pass;
        u32 common_overlap = l.ctlbit.overlap & r.ctlbit.overlap;
#pragma unroll
        for(u32 i = 0; i < 2; ++i)
        {
            u32 encode_home = (1 << arr[i]->ctlbit.home);
            if(arr[i]->ctlbit.home < pass && (common_overlap & encode_home))
            {
                //muda_kernel_printf("already, cid(%d,%d) home=(%d,%d) pass(%d,%d) eid(%d,%d) ijk{(%d,%d,%d), (%d,%d,%d)}\n",
                //                   l.cid,
                //                   r.cid,
                //                   l.ctlbit.home,
                //                   r.ctlbit.home,
                //                   l.ctlbit.pass,
                //                   r.ctlbit.pass,
                //                   l.oid,
                //                   r.oid,
                //                   l.ijk(0),
                //                   l.ijk(1),
                //                   l.ijk(2),
                //                   r.ijk(0),
                //                   r.ijk(1),
                //                   r.ijk(2));
                return true;
            }
        }
        //muda_kernel_printf("valid, cid(%d,%d) home=(%d,%d) pass(%d,%d) eid(%d,%d) ijk{(%d,%d,%d), (%d,%d,%d)}\n",
        //                   l.cid,
        //                   r.cid,
        //                   l.ctlbit.home,
        //                   r.ctlbit.home,
        //                   l.ctlbit.pass,
        //                   r.ctlbit.pass,
        //                   l.oid,
        //                   r.oid,
        //                   l.ijk(0),
        //                   l.ijk(1),
        //                   l.ijk(2),
        //                   r.ijk(0),
        //                   r.ijk(1),
        //                   r.ijk(2));
        return false;
    }

    static void csvHeader(std::ostream& os)
    {
        os << "cid,oid,pass,home,overlap,i,j,k" << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os, const SpatialPartitionCell& cell)
    {
        os << std::hex << cell.cid << "," << std::dec << cell.oid << "," << std::hex
           << cell.ctlbit.pass << "," << cell.ctlbit.home << "," << cell.ctlbit.overlap
           << "," << std::dec << cell.ijk(0) << "," << cell.ijk(1) << "," << cell.ijk(2);
        return os;
    }
};

template <typename Hash = muda::shift_hash<20, 10, 0>>
class SpatialHashConfig
{
    using real      = float;
    using u32       = uint32_t;
    using vec3      = Eigen::Vector<real, 3>;
    using ivec3     = Eigen::Vector<int, 3>;
    using uvec2     = Eigen::Vector<u32, 2>;
    using uvec3     = Eigen::Vector<u32, 3>;
    using hash_func = Hash;

  public:
    float            cellSize = 0.0f;
    vec3             coordMin;
    MUDA_GENERIC u32 hashCell(const vec3& xyz) const
    {
        return hashCell(cell(xyz));
    }

    MUDA_GENERIC u32 hashCell(const ivec3& ijk) const
    {
        return hash_func()(ijk) % 0x40000000;
    }

    MUDA_GENERIC ivec3 cell(const vec3& xyz) const
    {
        ivec3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = (xyz(i) - coordMin(i)) / cellSize;
        return ret;
    }
    MUDA_GENERIC vec3 coord(const ivec3& ijk) const
    {
        vec3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = ijk(i) * cellSize + coordMin(i);
        return ret;
    }
    MUDA_GENERIC vec3 cellCenterCoord(const ivec3& ijk) const
    {
        vec3 ret;
#pragma unroll
        for(int i = 0; i < 3; ++i)
            ret(i) = (ijk(i) + 0.5f) * cellSize + coordMin(i);
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

template <typename Hash>
class SpatialPartitionLauncher;

namespace details
{
    template <typename Hash = morton>
    class SpatialPartitionField
    {
      public:
        template <typename T>
        using dvec = device_buffer<T>;

        using Cell      = SpatialPartitionCell;
        using hash_type = Hash;

        cudaStream_t    m_stream;
        int             lightKernelBlockDim;
        int             heavyKernelBlockDim;
        dense1D<sphere> spheres;


        //float           cellSize_;
        device_var<int>   cellCount;
        device_var<int>   pairCount;
        device_var<float> maxRadius;

        device_var<SpatialHashConfig<hash_type>> spatialHashConfig;
        SpatialHashConfig<hash_type>             h_spatialHashConfig;

        dvec<SpatialPartitionCell> cellArrayValue;
        dvec<SpatialPartitionCell> cellArrayValueSorted;
        dvec<int>                  cellArrayKey;
        dvec<int>                  cellArrayKeySorted;

        dvec<int>       uniqueKey;
        device_var<int> uniqueKeyCount;
        int             validCellCount;
        int             sum;

        dvec<int> objCountInCell;
        dvec<int> objCountInCellPrefixSum;

        dvec<int> collisionPairCount;
        dvec<int> collisionPairPrefixSum;

        //using hash_type = Hash;
        SpatialPartitionField() = default;

        void configLaunch(int lightKernelBlockDim, int heavyKernelBlockDim)
        {
            this->lightKernelBlockDim = lightKernelBlockDim;
            this->heavyKernelBlockDim = heavyKernelBlockDim;
        }

        void configSpatialHash(const Eigen::Vector3f& coordMin, float cellSize = -1.0f)
        {
            h_spatialHashConfig.coordMin = coordMin;
            h_spatialHashConfig.cellSize = cellSize;
            spatialHashConfig            = h_spatialHashConfig;
        }

        template <typename Pred>
        void beginCreateCollisionPairs(dense1D<sphere> boundingSphereList,
                                       device_buffer<CollisionPair>& collisionPairs,
                                       Pred&&                        pred)
        {
            spheres = boundingSphereList;

            if(h_spatialHashConfig.cellSize <= 0.0f)  // to calculate the bounding sphere
                beginCalculateCellSize();

            beginSetupHashTable();
            waitAndCreateTempData();

            beginCountCollisionPairs(std::forward<Pred>(pred));
            waitAndAllocCollisionPairList(collisionPairs);

            beginSetupCollisionPairList(collisionPairs, std::forward<Pred>(pred));
        }

        device_buffer<float>     allRadius;
        device_buffer<std::byte> cellSizeBuf;

        void beginSetupHashTable()
        {
            beginFillHashCells();
            beginSortHashCells();
            beginCountCollisionPerCell();
        }

        void beginCalculateCellSize();

        void beginFillHashCells();

        device_buffer<std::byte> cellArraySortBuf;

        void beginSortHashCells();

        device_buffer<std::byte> encodeBuf;

        void beginCountCollisionPerCell();

        void waitAndCreateTempData();

        template <typename Pred>
        void beginCountCollisionPairs(Pred&& pred);

        device_buffer<std::byte> collisionScanBuf;

        void waitAndAllocCollisionPairList(device_buffer<CollisionPair>& collisionPairs);

        template <typename Pred>
        void beginSetupCollisionPairList(device_buffer<CollisionPair>& collisionPairs,
                                         Pred&&                        pred);

        void stream(cudaStream_t stream)
        {
            m_stream = stream;
            cellArrayValue.stream(stream);

            cellArrayKey.stream(stream);
            cellArrayValueSorted.stream(stream);
            cellArrayKeySorted.stream(stream);

            uniqueKey.stream(stream);

            objCountInCell.stream(stream);
            objCountInCellPrefixSum.stream(stream);

            collisionPairCount.stream(stream);
            collisionPairPrefixSum.stream(stream);

            allRadius.stream(stream);
            cellSizeBuf.stream(stream);
        }
    };
}  // namespace details

template <typename Hash = morton>
class SpatialPartitionField
{
    details::SpatialPartitionField<Hash> m_impl;

  public:
    template <typename Hash>
    friend class SpatialPartitionLauncher;
};


//usage:
//  stream s;
//  SpatialPartitionField field;
//  device_buffer<CollisionPair> res;
//  device_buffer<sphere> spheres;
//
//  // one-step API
//  on(s)
//      .next<SpatialPartitionLauncher>(field)
//      .applyCreateCollisionPairs(spheres, res,
//          []__device__(int i, int j) // optional
//          {
//              print("we find (%f,%f)!", i, j);
//              return true;
//          })
//      .wait(); // wait to get collision pairs, or do something on the same stream (this time implicit sync happens)
//
//  // advanced API
//  on(s)
//      .next<SpatialPartitionLauncher>(field)
//      .applyCalculateCellSize()
//      .applySetupHashTable()
//      .waitAndCreateTempData()
//      .applyCountCollisionPairs()
//      .waitAndAllocCollisionPairList(res)
//      .applySetupCollisionPairList(res)
//      .wait();
//
// about <Predicate> : bool (int id_left, int id_right)
template <typename Hash = morton>
class SpatialPartitionLauncher : public launch_base<SpatialPartitionLauncher<Hash>>
{
    // algorithm comes from:
    // https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  private:
    using Field     = SpatialPartitionField<Hash>;
    using FieldImpl = details::SpatialPartitionField<Hash>;

    FieldImpl& m_field;
    int        lightKernelBlockDim;
    int        heavyKernelBlockDim;

  public:
    class DefaultPred
    {
      public:
        __device__ bool operator()(int i, int j) { return true; }
    };

    SpatialPartitionLauncher(Field&       field,
                             int          lightKernelBlockDim = 256,
                             int          heavyKernelBlockDim = 64,
                             cudaStream_t stream              = nullptr)
        : launch_base<SpatialPartitionLauncher<Hash>>(stream)
        , m_field(field.m_impl)
    {
        m_field.stream(stream);
        m_field.configLaunch(lightKernelBlockDim, heavyKernelBlockDim);
    }

    virtual void init_stream(cudaStream_t s) override { m_field.stream(s); }

    SpatialPartitionLauncher& configSpatialHash(const Eigen::Vector3f& coordMin,
                                                float cellSize = -1.0f)
    {
        m_field.configSpatialHash(coordMin, cellSize);
        return *this;
    }

    /// <summary>
    /// one-step API, create collision pairs from bounding spheres, some host waits exist.
    /// </summary>
    /// <param name="boundingSphereList"></param>
    /// <param name="collisionPairs"></param>
    /// <returns></returns>

    template <typename Pred = DefaultPred>
    SpatialPartitionLauncher& applyCreateCollisionPairs(dense1D<sphere> boundingSphereList,
                                                        device_buffer<CollisionPair>& collisionPairs,
                                                        Pred&& pred = {})
    {
        m_field.beginCreateCollisionPairs(
            boundingSphereList, collisionPairs, std::forward<Pred>(pred));
        return *this;
    }

    SpatialPartitionLauncher& applyCalculateCellSize()
    {
        m_field.beginCalculateCellSize();
        return *this;
    }

    SpatialPartitionLauncher& applySetupHashTable()
    {
        m_field.beginSetupHashTable();
        return *this;
    }

    SpatialPartitionLauncher& waitAndCreateTempData()
    {
        m_field.waitAndCreateTempData();
        return *this;
    }

    template <typename Pred = DefaultPred>
    SpatialPartitionLauncher& applyCountCollisionPairs(Pred&& pred = {})
    {
        m_field.beginCountCollisionPairs(std::forward<Pred>(pred));
        return *this;
    }

    SpatialPartitionLauncher& waitAndAllocCollisionPairList(device_buffer<CollisionPair>& collisionPairs)
    {
        m_field.waitAndAllocCollisionPairList(collisionPairs);
        return *this;
    }

    template <typename Pred = DefaultPred>
    SpatialPartitionLauncher& applySetupCollisionPairList(device_buffer<CollisionPair>& collisionPairs,
                                                          Pred&& pred = {})
    {
        m_field.beginSetupCollisionPairList(collisionPairs, std::forward<Pred>(pred));
        return *this;
    }
};
}  // namespace muda

#include "spatial_hash.inl"