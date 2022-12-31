#pragma once
#include "../../muda_def.h"
#include "../../launch/launch_base.h"
#include "../../encode/morton.h"
#include "../../viewer/idxer.h"

#include "../../launch/parallel_for.h"
#include "../../launch/launch.h"
#include "../../buffer/device_buffer.h"
#include "../../container.h"

#include "bounding_volume.h"
#include "collide.h"
#include "../../algorithm/device_reduce.h"
#include "../../algorithm/device_radix_sort.h"
#include "../../algorithm/device_run_length_encode.h"
#include "../../algorithm/device_scan.h"
#include "../../encode/hash.h"
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
        {
            return true;
        }

        if(l.isPhantom() && r.isPhantom())
        {
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
                return true;
            }
        }
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
           << "," << cell.ijk(0) << "," << cell.ijk(1) << "," << cell.ijk(2);
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
    float            cellSize = 0.2f;
    vec3             coordMin;
    MUDA_GENERIC u32 hashCell(const vec3& xyz) const
    {
        return hashCell(cell(xyz));
    }

    MUDA_GENERIC u32 hashCell(const ivec3& ijk) const
    {
        return muda::shift_hash<20, 10, 0>::map(ijk) % 0x40000000;
        //return hash(uvec3(ijk.x(), ijk.y(), ijk.z())) % 0x40000000;
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
    MUDA_GENERIC    CollisionPair() = default;

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



/// <summary>
/// Using Spatial Partition to do broad phase collision detection
/// usage:
///     SpatialPartition sp;
///     sp.configSpatialHash(Vector3f::Zero());
///     sp
///         .beginSetupSpatialDataStructure(spheres)
///         .beginCreateCollisionPairList();
///     auto& peres = sp.waitToGetCollisionPairs()
/// </summary>
/// <typeparam name="Hash"></typeparam>
template <typename Hash = morton>
class SpatialPartition : public launch_base<SpatialPartition<Hash>>
{
	// algorithm comes from:
    // https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
  public:
    template <typename T>
    using dvec = device_buffer<T>;

    using Cell = SpatialPartitionCell;
	using hash_type = Hash;

    SpatialPartitionLaunchInfo launchInfo;
    idxer1D<sphere>            spheres;


    //float           cellSize_;
    device_var<int>   cellCount;
    device_var<int>   pairCount;
    device_var<float> maxRadius;

    dvec<SpatialPartitionCell> cellArrayValue;
    dvec<SpatialPartitionCell> cellArrayValueSorted;
    dvec<int>                  cellArrayKey;
    dvec<int>                  cellArrayKeySorted;

    dvec<int>       uniqueKey;
    device_var<int> uniqueKeyCount;
    int             validCellCount;

    dvec<int> objCountInCell;
    dvec<int> objCountInCellPrefixSum;

    dvec<int> collisionPairCount;
    dvec<int> collisionPairPrefixSum;

    SpatialHashConfig<hash_type> spatialHashConfig;

  public:
    //using hash_type = Hash;
    [[nodiscard]] SpatialPartition(cudaStream_t stream = nullptr)
        : launch_base<SpatialPartition<Hash>>(stream)
    {
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
        collisionPairs.stream(stream);
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

    SpatialPartition& beginSetupSpatialDataStructure(idxer1D<sphere> boundingSphereList)
    {
        spheres = boundingSphereList;

        if(spatialHashConfig.cellSize <= 0.0f)  // to calculate the bounding sphere
            beginCalculateCellSize();

        beginFillHashCells();
        beginSortHashCells();
        beginCountCollisionPerCell();
        return *this;
    }

    device_buffer<float>     allRadius;
    device_buffer<std::byte> cellSizeBuf;

    SpatialPartition& beginCalculateCellSize()
    {
        auto count = spheres.total_size();
        allRadius.resize(count);
        launch::wait_device();
        parallel_for(launchInfo.lightKernelBlockDim, 0, stream_)
            .apply(count,
                   [spheres = this->spheres, allRadius = make_viewer(allRadius)] __device__(
                       int i) mutable { allRadius(i) = spheres(i).r; })

            .next(DeviceReduce(stream_))
            .Max(cellSizeBuf, data(maxRadius), allRadius.data(), count)
            .wait();  

        // wait for maxRadius

        float r = maxRadius;
		
        // Scaling the bounding sphere of each object by sqrt(2) [we will use proxy spheres]
        // and ensuring that the grid cell is at least 1.5 times
        // as large as the scaled bounding sphere of the largest object.
        //https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
        spatialHashConfig.cellSize = r * 1.5f * 1.5f;
        return *this;
    }

    SpatialPartition& setCellSize(float cellSize)
    {
        spatialHashConfig.cellSize = cellSize;
        return *this;
    }

    SpatialPartition& beginFillHashCells()
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

            collisionPairCount.resize(count, buf_op::ignore);
            collisionPairPrefixSum.resize(count, buf_op::ignore);
        }

        parallel_for(launchInfo.lightKernelBlockDim, 0, stream_)
            .apply(spheres.total_size(),
                   [spheres        = this->spheres,
                    sh             = this->spatialHashConfig,
                    cellArrayValue = make_idxer2D(cellArrayValue, size, 8),
                    cellArrayKey = make_idxer2D(cellArrayKey, size, 8)] __device__(int i) mutable
                   {
                       using ivec3 = Eigen::Vector3i;
                       using vec3  = Eigen::Vector3f;
                       using u32   = uint32_t;

                       sphere s           = spheres(i);
                       auto   proxySphere = s;
                       //https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
                       // Scaling the bounding sphere of each object by sqrt(2), here we take 1.5(>1.414)
                       proxySphere.r *= 1.5;

                       auto  o        = s.o;
                       ivec3 ijk      = sh.cell(o);
                       auto  hash     = sh.hashCell(ijk);
                       auto  cellSize = sh.cellSize;

                       auto objectId = i;


                       Cell homeCell(hash, objectId);

                       // ...[i*8+0][i*8+1][i*8+2][i*8+3][i*8+4][i*8+5][i*8+6][i*8+7]...
                       // ...[hcell][pcell][pcell] ...   [  x  ][  x  ][  x  ][  x  ]...


                       // fill the cell that contains the center of the current sphere
                       homeCell.setAsHome(ijk);
                       homeCell.ijk = ijk;
                       vec3 xyz     = sh.cellCenterCoord(ijk);

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

                       int idx = 1;  //goes from 1 -> 7. idx = 0 is for the homeCell
#pragma unroll
                       for(int i = 0; i < 7; ++i)
                       {
                           vec3       min = sh.coord(cells[i]);
                           vec3       max = min + size;
                           muda::AABB aabb(min, max);

                           // use proxySphere to test
                           // whether the current sphere overlaps the neighbor cell
                           if(collide::detect(proxySphere, aabb))
                           {
                               homeCell.setOverlap(cells[i]);
                               auto hash = sh.hashCell(cells[i]);
                               Cell phantom(hash, objectId);
                               phantom.setAsPhantom(ijk, cells[i]);
                               phantom.ijk            = cells[i];
                               phantom.ctlbit.overlap = homeCell.ctlbit.overlap;
                               cellArrayValue(objectId, idx++) = phantom;
                           }
                       }

                       //set the home cell
                       cellArrayValue(objectId, 0) = homeCell;

                       // turn off othese non-overlap neighbor cells if we do have.
                       for(; idx < 8; ++idx)
                           cellArrayValue(objectId, idx) = Cell(-1, -1);

                       // fill the key for later sorting
                       for(int i = 0; i < 8; ++i)
                           cellArrayKey(objectId, i) =
                               cellArrayValue(objectId, i).cid;
                   });

        return *this;
    }

    device_buffer<std::byte> cellArraySortBuf;
    SpatialPartition&        beginSortHashCells()
    {
        DeviceRadixSort(stream_).SortPairs(cellArraySortBuf,
                                           (uint32_t*)cellArrayKeySorted.data(),  //out
                                           cellArrayValueSorted.data(),  //out
                                           (uint32_t*)cellArrayKey.data(),  //in
                                           cellArrayValue.data(),           //in
                                           spheres.total_size() * 8);
        ////debug output sorted cell array
        //host_vector<Cell> hcellArrayValueSorted;
        //cellArrayValueSorted.copy_to(hcellArrayValueSorted);
        //launch::wait_device();
        //csv(Cell::csvHeader, hcellArrayValueSorted, "hcellArrayValueSorted.csv");
        return *this;
    }

    device_buffer<std::byte> encodeBuf;
    SpatialPartition&        beginCountCollisionPerCell()
    {
        auto count = spheres.total_size() * 8;
        DeviceRunLengthEncode(stream_)
            .Encode(encodeBuf,
                    uniqueKey.data(),           // out
                    objCountInCell.data(),      // out
                    data(uniqueKeyCount),       // out
                    cellArrayKeySorted.data(),  // in
                    count)
            .wait();  
        
        // wait for uniqueKeyCount
        // we need use uniqueKeyCount to resize arrays

        int h_uniqueKeyCount = uniqueKeyCount;

        uniqueKey.resize(h_uniqueKeyCount);
        objCountInCell.resize(h_uniqueKeyCount);
        objCountInCellPrefixSum.resize(h_uniqueKeyCount);
        collisionPairCount.resize(h_uniqueKeyCount);
        collisionPairPrefixSum.resize(h_uniqueKeyCount);

        // validCellCount is always 1 less than the uniqueKeyCount, 
        // the last one is typically the cell-object-pair {cid = -1, oid = -1}
        validCellCount = h_uniqueKeyCount - 1;

        // we still prefix sum the uniqueKeyCount cell-object-pair
        // because we need to know the total number of collision pairs
        // which is at the last element of the prefix sum array
        DeviceScan(stream_).ExclusiveSum(cellArraySortBuf,
                                         objCountInCellPrefixSum.data(),
                                         objCountInCell.data(),
                                         h_uniqueKeyCount);
        return *this;
    }

    device_buffer<std::byte>     collisionScanBuf;
    device_buffer<CollisionPair> collisionPairs;
    SpatialPartition&            beginCreateCollisionPairList()
    {
        int sum;
        // the last one is {cid = uint32_t(-1), oid = uint32_t(-1)}
        int  count;
        auto lastOffset = uniqueKeyCount - 1;

        parallel_for(launchInfo.lightKernelBlockDim, 0, stream_)
            .apply(validCellCount,
                   [spheres        = this->spheres,
                    objCountInCell = make_viewer(objCountInCell),
                    objCountInCellPrefixSum = make_viewer(objCountInCellPrefixSum),
                    cellArrayValueSorted = make_viewer(cellArrayValueSorted),
                    collisionPairCount = make_viewer(collisionPairCount)] __device__(int cell) mutable
                   {
                       int size      = objCountInCell(cell);
                       int offset    = objCountInCellPrefixSum(cell);
                       int pairCount = 0;

                       for(int i = 0; i < size; ++i)
                       {
                           auto cell0 = cellArrayValueSorted(offset + i);
                           auto oid0  = cell0.oid;
                           for(int j = i + 1; j < size; ++j)
                           {
                               auto cell1 = cellArrayValueSorted(offset + j);
                               auto oid1  = cell1.oid;
                               //print("test => %d,%d\n", oid0, oid1);
                               if(!Cell::allowIgnore(cell0, cell1)  // cell0, cell1 are created by test the proxy sphere
                                  && collide::detect(spheres(oid0), spheres(oid1)))  // test the bounding spheres to get exact collision result
                               {
                                   //print("pair");
                                   ++pairCount;
                               }
                           }
                       }
                       collisionPairCount(cell) = pairCount;
                   })

            .next(DeviceScan(stream_))
            .ExclusiveSum(collisionScanBuf,
                          collisionPairPrefixSum.data(),
                          collisionPairCount.data(),
                          uniqueKeyCount)

            .next(memory(stream_))
            .copy(&sum, collisionPairPrefixSum.data() + lastOffset, sizeof(int), cudaMemcpyDeviceToHost)
            .wait();  // wait for sum

        //debug output
        //host_vector<int> hCollisionPairCount, hCollisionPairPrefixSum;
        //collisionPairCount.copy_to(hCollisionPairCount);
        //collisionPairPrefixSum.copy_to(hCollisionPairPrefixSum);

        //launch::wait_device();
        //csv(hCollisionPairCount, "hCollisionPairCount.csv");
        //csv(hCollisionPairPrefixSum, "hCollisionPairPrefixSum.csv");

        int totalCollisionPairCount = sum;
        collisionPairs.stream(stream_);
        collisionPairs.resize(totalCollisionPairCount);

        parallel_for(launchInfo.lightKernelBlockDim, 0, stream_)
            .apply(validCellCount,
                   [spheres        = this->spheres,
                    objCountInCell = make_viewer(objCountInCell),
                    objCountInCellPrefixSum = make_viewer(objCountInCellPrefixSum),
                    cellArrayValueSorted = make_viewer(cellArrayValueSorted),
                    collisionPairCount   = make_viewer(collisionPairCount),
                    collisionPairPrefixSum = make_viewer(collisionPairPrefixSum),
                    collisionPairs = make_viewer(collisionPairs)] __device__(int cell) mutable
                   {
                       int size       = objCountInCell(cell);
                       int offset     = objCountInCellPrefixSum(cell);
                       int pairOffset = collisionPairPrefixSum(cell);
                       int index      = 0;
                       for(int i = 0; i < size; ++i)
                       {
                           auto cell0 = cellArrayValueSorted(offset + i);
                           auto oid0  = cell0.oid;
                           for(int j = i + 1; j < size; ++j)
                           {
                               auto cell1 = cellArrayValueSorted(offset + j);
                               auto oid1  = cell1.oid;
                               if(!Cell::allowIgnore(cell0, cell1)
                                  && collide::detect(spheres(oid0), spheres(oid1)))
                               {
                                   CollisionPair p;
                                   p.id[0]                            = oid0;
                                   p.id[1]                            = oid1;
                                   collisionPairs(pairOffset + index) = p;
                                   ++index;
                               }
                           }
                       }
                   });

        return *this;
    }

    device_buffer<CollisionPair>& waitToGetCollisionPairs()
    {
        launch::wait_stream(stream_);
        return collisionPairs;
    };
    device_buffer<CollisionPair>& getCollisionPairs() { return collisionPairs; }
    template <typename F>
    SpatialPartition& beginApplyOnEachCollisionPair(int nonemptyCellCount, F&& func)
    {
        using CallableType = raw_type_t<F>;
        static_assert(std::is_invocable_v<CallableType, int, int>, "f:void(int i, int j)");
        parallel_for(launchInfo.lightKernelBlockDim, 0, stream_)
            .apply(nonemptyCellCount,
                   [func           = func,
                    objCountInCell = make_viewer(objCountInCell),
                    objCountInCellPrefixSum = make_viewer(objCountInCellPrefixSum),
                    cellArrayValueSorted = make_viewer(cellArrayValueSorted),
                    collisionPairCount = make_viewer(collisionPairCount)] __device__(int cell) mutable
                   {
                       int size      = objCountInCell(cell);
                       int offset    = objCountInCellPrefixSum(cell);
                       int pairCount = 0;

                       for(int i = 0; i < size; ++i)
                       {
                           auto cell0 = cellArrayValueSorted(offset + i);
                           auto e0    = cell0.oid;
                           for(int j = i + 1; j < size; ++j)
                           {
                               auto cell1 = cellArrayValueSorted(offset + j);
                               auto e1    = cell1.oid;
                               if(!Cell::allowIgnore(cell0, cell1))
                               {
                                   func(i, j);
                               }
                           }
                       }
                   });
        return *this;
    }
};
}  // namespace muda