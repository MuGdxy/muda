#pragma once
#include "../../muda_def.h"
#include "../../launch/launch_base.h"
#include "../../encode/morton.h"
#include "../../viewer/idxer.h"
#include "bounding_volume.h"
#include "../../launch/parallel_for.h"
#include "../../buffer/device_buffer.h"
#include "../../algorithm/reduce.h"
#include "../../algorithm/radix_sort.h"
#include "../../container.h"
#include "collide.h"


namespace muda
{
struct SpatialPartitionLaunchInfo
{
    int lightKernelBlockDim = 256;
    int heavyKernelBlockDim = 64;
};


class SpatialPartitionCell
{
  public:
    using ivec3 = Eigen::Vector3<int>;
    using u32   = uint32_t;
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
    struct
    {
        u32 pass : 3;
        u32 home : 3;
        u32 overlap : 8;
    } ctlbit;

    u32   cid;  // cell id
    u32   oid;  // object id (edge id)
    ivec3 ijk;  // debug only
    bool  home;

    MUDA_GENERIC static bool allowIgnore(const SpatialPartitionCell& l,
                                         const SpatialPartitionCell& r)
    {
        if(l.ijk != r.ijk)
            return true;
        if(l.isPhantom() && r.isPhantom())
            return true;
        const SpatialPartitionCell* arr[] = {&l, &r};

        u32 pass           = l.ctlbit.pass;
        u32 common_overlap = l.ctlbit.overlap & r.ctlbit.overlap;
#pragma unroll
        for(u32 i = 0; i < 2; ++i)
        {
            u32 encode_home = (1 << arr[i]->ctlbit.home);
            if(arr[i]->ctlbit.home < pass && (common_overlap & encode_home))
                return true;
        }
        return false;
    }
};

template <typename Hash = muda::morton>
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
    float            cellSize = 0.2f;
    vec3             coordMin;
    hash_func        hash;
    MUDA_GENERIC u32 hashCell(const vec3& xyz) const
    {
        return hashCell(cell(xyz));
    }

    MUDA_GENERIC u32 hashCell(const ivec3& ijk) const
    {
        return hash(uvec3(ijk.x(),ijk.y(),ijk.z())) % 0x40000000;
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
    uint32_t id[2];
};

//https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
//template <typename Hash = morton>
class SpatialPartition : public launch_base<SpatialPartition>
{
    using hash = morton;

    template <typename T>
    using dvec = device_buffer<T>;

    using Cell = SpatialPartitionCell;

    SpatialPartitionLaunchInfo launchInfo;
    idxer1D<sphere>            spheres;

    float           cellSize_;
    device_var<int> cellCount;
    device_var<int> pairCount;

    dvec<SpatialPartitionCell> cellArrayValue;
    dvec<SpatialPartitionCell> cellArrayValueSorted;
    dvec<int>                  cellArrayKey;
    dvec<int>                  cellArrayKeySorted;

    dvec<int> uniqueKey;

    dvec<int> objCountInCell;
    dvec<int> objCountInCellPrefixSum;

    dvec<int> collisionPairCount;
    dvec<int> collisionPairPrefixSum;

    SpatialHashConfig<hash> spatialHashConfig;

  public:
    //using hash_type = Hash;
    [[nodiscard]] SpatialPartition(cudaStream_t stream = nullptr)
        : launch_base(stream)
    {
    }

    SpatialPartition& configLaunch(SpatialPartitionLaunchInfo info)
    {
        launchInfo = info;
        return *this;
    }

    SpatialPartition& configSpatialHash(const Eigen::Vector3f coordMin)
    {
        spatialHashConfig.coordMin = coordMin;
        return *this;
    }

    SpatialPartition& prepare(idxer1D<sphere> boundingSphereList) 
    {
        spheres = boundingSphereList;
        calculateCellSize();
        fillHashCells();
        sortHashCells();
        runLengthEncodeHashCells();
        countCollisionPerCell();
        return *this;
    }

    SpatialPartition& calculateCellSize()
    {
        
        device_buffer         buf;
        device_buffer<sphere> maxRadiusShpere(stream_, 1);

        // to get the max radius
        DeviceReduce(stream_).Reduce(
            buf,
            maxRadiusShpere.data(),
            spheres.data(),
            spheres.total_size(),
            [] __device__(const sphere& a, const sphere& b)
            { return a.r > b.r ? a : b; },
            sphere{});

        sphere s;
        maxRadiusShpere.copy_to(s);
        cellSize_ = s.r * 2.5f;
        return *this;
    }

    SpatialPartition& setCellSize(int cellSize)
    {
        cellSize_ = cellSize;
        return *this;
    }

    SpatialPartition& fillHashCells()
    {
        int size  = spheres.total_size();
        int count = 8 * size;
        if(cellArrayValue.size() < count)
        {
            cellArrayValue.resize(count, buf_op::ignore);

            cellArrayKey.resize(count, buf_op::ignore);
            cellArrayValueSorted.resize(count, buf_op::ignore);
            cellArrayKeySorted.resize(count, buf_op::ignore);

            uniqueKey.resize(count, buf_op::ignore);

            objCountInCell.resize(count, buf_op::ignore);
            objCountInCellPrefixSum.resize(count, buf_op::ignore);
        }

        parallel_for(launchInfo.lightKernelBlockDim)
            .apply(spheres.total_size(),
                   [spheres        = this->spheres,
                    sh             = this->spatialHashConfig,
                    cellSize       = this->cellSize_,
                    cellArrayValue = make_idxer2D(cellArrayValue, size, 8),
                    cellArrayKey = make_idxer2D(cellArrayKey, size, 8)] __device__(int i) mutable
                   {
                       using ivec3 = Eigen::Vector3i;
                       using vec3  = Eigen::Vector3f;
                       using u32   = uint32_t;

                       sphere s = spheres(i);

                       auto  o    = s.o;
                       ivec3 ijk  = sh.cell(o);
                       auto  hash = sh.hashCell(ijk);
                       Cell  homeCell(hash, s.id);

                       //goes from 0 -> 7.
                       int idx = 0;

                       // ...[i*8+0][i*8+1][i*8+2][i*8+3][i*8+4][i*8+5][i*8+6][i*8+7]...
                       // ...[hcell][pcell][pcell] ...   [  x  ][  x  ][  x  ][  x  ]...


                       // fill the cell that contains the center of the current sphere
                       homeCell.setAsHome(ijk);
                       homeCell.ijk                = ijk;
                       vec3 xyz                    = sh.cellCenterCoord(ijk);
                       cellArrayValue(s.id, idx++) = homeCell;

                       //find the cloest 7 neighbor cells
                       ivec3 dxyz;
#pragma unroll
                       for(int i = 0; i < 3; ++i)
                           dxyz(i) = o(i) > xyz(i) ? 1 : -1;

                       ivec3 cells[7] = {ijk + ivec3(dxyz.x(), 0, 0),
                                         ijk + ivec3(0, dxyz.y(), 0),
                                         ijk + ivec3(0, 0, dxyz.z()),
                                         ijk + ivec3(0, dxyz.y(), dxyz.z()),
                                         ijk + ivec3(dxyz.x(), 0, dxyz.z()),
                                         ijk + ivec3(dxyz.x(), dxyz.y(), 0),
                                         ijk + dxyz};

                       // the cell size (3d)
                       vec3 size(cellSize, cellSize, cellSize);

#pragma unroll
                       for(int i = 0; i < 7; ++i)
                       {
                           vec3       min = sh.coord(cells[i]);
                           vec3       max = min + size;
                           muda::AABB aabb(min, max);
                           // whether the current sphere overlaps the neighbor cell
                           if(collide::detect(s, aabb))
                           {
                               homeCell.setOverlap(cells[i]);

                               Cell phantom(sh.hashCell(cells[i]), s.id);
                               phantom.setAsPhantom(ijk, cells[i]);
                               phantom.ijk            = cells[i];
                               phantom.ctlbit.overlap = homeCell.ctlbit.overlap;
                               cellArrayValue(s.id, idx++) = phantom;
                           }
                       }

                       // turn off othese non-overlap neighbor cells if we do have.
                       for(; idx < 8; ++idx)
                           cellArrayValue(s.id, idx) = Cell(-1, -1);

                       // fill the key for later sorting
                       for(int i = 0; i < 8; ++i)
                           cellArrayKey(s.id, i) = cellArrayValue(s.id, i).cid;
                   });
        return *this;
    }

    device_buffer<std::byte> cellArraySortBuf;
    SpatialPartition& sortHashCells() 
    {
        DeviceRadixSort(stream_).SortPairs(cellArraySortBuf,
                                           cellArrayKeySorted.data(),
                                           cellArrayKey.data(),
                                           cellArrayValue.data(),
                                           cellArrayValueSorted.data(),
                                           spheres.total_size() * 8);
        return *this;
    }

    SpatialPartition& runLengthEncodeHashCells() { return *this; }
    SpatialPartition& countCollisionPerCell() { return *this; }
    SpatialPartition& createCollisionPairList() { return *this; }
    SpatialPartition& applyOnEachCollisionPair() { return *this; }
};
}  // namespace muda