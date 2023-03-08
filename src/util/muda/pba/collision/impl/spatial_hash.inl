#ifdef __INTELLISENSE__
#include "../spatial_hash.h"
#endif

namespace muda::details
{
template <typename Hash>
inline void SpatialPartitionField<Hash>::configLaunch(int lightKernelBlockDim, int heavyKernelBlockDim)
{
    this->lightKernelBlockDim = lightKernelBlockDim;
    this->heavyKernelBlockDim = heavyKernelBlockDim;
}

template <typename Hash>
inline void SpatialPartitionField<Hash>::configSpatialHash(const Eigen::Vector3f& coordMin,
                                                           float cellSize)
{
    h_spatialHashConfig.coordMin = coordMin;
    h_spatialHashConfig.cellSize = cellSize;
    spatialHashConfig            = h_spatialHashConfig;
}

template <typename Hash>
inline void SpatialPartitionField<Hash>::beginSetupHashTable()
{
    beginFillHashCells();
    beginSortHashCells();
    beginCountCollisionPerCell();
}

template <typename Hash>
template <typename Pred>
inline void SpatialPartitionField<Hash>::beginCreateCollisionPairs(
    dense1D<sphere> boundingSphereList, device_buffer<CollisionPair>& collisionPairs, Pred&& pred)
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

template <typename Hash>
inline void SpatialPartitionField<Hash>::stream(cudaStream_t stream)
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

template <typename Hash>
void SpatialPartitionField<Hash>::beginCalculateCellSize()
{
    auto count = spheres.total_size();
    allRadius.resize(count);

    parallel_for(lightKernelBlockDim, 0, m_stream)
        .apply(count,
               [spheres = this->spheres, allRadius = make_viewer(allRadius)] __device__(
                   int i) mutable { allRadius(i) = spheres(i).r; });

    DeviceReduce(m_stream).Max(cellSizeBuf, data(maxRadius), allRadius.data(), count);

    launch(1, 1, 0, m_stream)
        .apply(
            [spatialHashConfig = make_viewer(spatialHashConfig),
             maxRadius         = make_viewer(maxRadius)] __device__() mutable
            {
                // Scaling the bounding sphere of each object by sqrt(2) [we will use proxy spheres]
                // and ensuring that the grid cell is at least 1.5 times
                // as large as the scaled bounding sphere of the largest object.
                //https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
                spatialHashConfig->cellSize = maxRadius * 2.25f;
                // print("spatialHashConfig->cellSize=%f\n", spatialHashConfig->cellSize);
            });
}

template <typename Hash>
void SpatialPartitionField<Hash>::beginFillHashCells()
{
    int size  = spheres.total_size();
    int count = 8 * size + 1;  //

    cellArrayValue.resize(count, buf_op::ignore);

    cellArrayKey.resize(count, buf_op::ignore);
    cellArrayValueSorted.resize(count, buf_op::ignore);
    cellArrayKeySorted.resize(count, buf_op::ignore);

    uniqueKey.resize(count, buf_op::ignore);

    objCountInCell.resize(count, buf_op::ignore);
    objCountInCellPrefixSum.resize(count, buf_op::ignore);

    collisionPairCount.resize(count, buf_op::ignore);
    collisionPairPrefixSum.resize(count, buf_op::ignore);

    parallel_for(lightKernelBlockDim, 0, m_stream)
        .apply(spheres.total_size(),
               [spheres           = spheres,
                spatialHashConfig = make_viewer(spatialHashConfig),
                cellArrayValue    = make_dense2D(cellArrayValue, size, 8),
                cellArrayKey = make_dense2D(cellArrayKey, size, 8)] __device__(int i) mutable
               {
                   using ivec3 = Eigen::Vector3i;
                   using vec3  = Eigen::Vector3f;
                   using u32   = uint32_t;

                   sphere s  = spheres(i);
                   auto&  sh = *spatialHashConfig;

                   auto proxySphere = s;
                   //https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
                   // Scaling the bounding sphere of each object by sqrt(2), here we take 1.5(>1.414)
                   proxySphere.r *= 1.5;

                   auto  o        = s.o;
                   ivec3 ijk      = sh.cell(o);
                   auto  hash     = sh.hashCell(ijk);
                   auto  cellSize = sh.cellSize;
                   // print("beginFillHashCells cellSize=%f\n", cellSize);
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
                       cellArrayKey(objectId, i) = cellArrayValue(objectId, i).cid;
               });

    launch(1, 1, 0, m_stream)
        .apply(
            [cellArrayValue = make_viewer(cellArrayValue),
             cellArrayKey   = make_viewer(cellArrayKey)] __device__() mutable
            {
                cellArrayKey(cellArrayKey.total_size() - 1)     = -1;
                cellArrayValue(cellArrayValue.total_size() - 1) = Cell(-1, -1);
            });
}

template <typename Hash>
void SpatialPartitionField<Hash>::beginSortHashCells()
{
    DeviceRadixSort(m_stream).SortPairs(cellArraySortBuf,
                                        (uint32_t*)cellArrayKeySorted.data(),  //out
                                        cellArrayValueSorted.data(),     //out
                                        (uint32_t*)cellArrayKey.data(),  //in
                                        cellArrayValue.data(),           //in
                                        cellArrayValue.size());
}

template <typename Hash>
void SpatialPartitionField<Hash>::beginCountCollisionPerCell()
{
    auto count = uniqueKey.size();
    DeviceRunLengthEncode(m_stream).Encode(encodeBuf,
                                           uniqueKey.data(),           // out
                                           objCountInCell.data(),      // out
                                           data(uniqueKeyCount),       // out
                                           cellArrayKeySorted.data(),  // in
                                           count);

    //.wait();

    //host_vector<SpatialPartitionCell> h;
    //cellArrayValueSorted.copy_to(h).wait();
    //std::vector<SpatialPartitionCell> v(h.begin(), h.end());
    //for(size_t i = 0; i < v.size(); i++)
    //{
    //    std::cout << "home " << std::hex << v[i].cid << std::dec << " "
    //              << v[i].ctlbit.home << "," << v[i].ctlbit.pass << std::endl;
    //}
}

template <typename Hash>
void SpatialPartitionField<Hash>::waitAndCreateTempData()
{
    on(m_stream).wait();

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
}

template <typename Hash>
template <typename Pred>
void SpatialPartitionField<Hash>::beginCountCollisionPairs(Pred&& pred)
{
    using CallableType = raw_type_t<Pred>;
    // we still prefix sum the uniqueKeyCount cell-object-pair
    // because we need to know the total number of collision pairs
    // which is at the last element of the prefix sum array
    DeviceScan(m_stream).ExclusiveSum(cellArraySortBuf,
                                      objCountInCellPrefixSum.data(),
                                      objCountInCell.data(),
                                      validCellCount + 1);

    parallel_for(lightKernelBlockDim, 0, m_stream)
        .apply(validCellCount,
               [spheres                 = this->spheres,
                objCountInCell          = make_viewer(objCountInCell),
                objCountInCellPrefixSum = make_viewer(objCountInCellPrefixSum),
                cellArrayValueSorted    = make_viewer(cellArrayValueSorted),
                collisionPairCount      = make_viewer(collisionPairCount),
                pred = std::forward<Pred>(pred)] __device__(int cell) mutable
               {
                   int size      = objCountInCell(cell);
                   int offset    = objCountInCellPrefixSum(cell);
                   int pairCount = 0;

                   for(int i = 0; i < size; ++i)
                   {
                       auto cell0 = cellArrayValueSorted(offset + i);
                       auto oid0  = cell0.oid;
                       auto s0    = spheres(oid0);
                       for(int j = i + 1; j < size; ++j)
                       {
                           auto cell1 = cellArrayValueSorted(offset + j);
                           auto oid1  = cell1.oid;
                           auto s1    = spheres(oid1);
                           if(!Cell::allowIgnore(cell0, cell1)
                              && collide::detect(s0, s1) && pred(s0.id, s1.id))  // user predicate
                           {
                               //print("pair");
                               ++pairCount;
                           }
                       }
                   }
                   collisionPairCount(cell) = pairCount;
               });

    int keycount = uniqueKeyCount;
    if(keycount)
    {
        DeviceScan(m_stream).ExclusiveSum(collisionScanBuf,
                                          collisionPairPrefixSum.data(),
                                          collisionPairCount.data(),
                                          keycount);
        auto lastOffset = validCellCount;
        memory(m_stream).copy(&sum,
                              collisionPairPrefixSum.data() + lastOffset,
                              sizeof(int),
                              cudaMemcpyDeviceToHost);
    }
    else
        sum = 0;
}

template <typename Hash>
void SpatialPartitionField<Hash>::waitAndAllocCollisionPairList(device_buffer<CollisionPair>& collisionPairs)
{
    on(m_stream).wait();

    int totalCollisionPairCount = sum;
    collisionPairs.stream(m_stream);
    collisionPairs.resize(totalCollisionPairCount);
}

template <typename Hash>
template <typename Pred>
void SpatialPartitionField<Hash>::beginSetupCollisionPairList(device_buffer<CollisionPair>& collisionPairs,
                                                              Pred&& pred)
{
    parallel_for(lightKernelBlockDim, 0, m_stream)
        .apply(validCellCount,
               [spheres                 = this->spheres,
                objCountInCell          = make_viewer(objCountInCell),
                objCountInCellPrefixSum = make_viewer(objCountInCellPrefixSum),
                cellArrayValueSorted    = make_viewer(cellArrayValueSorted),
                collisionPairCount      = make_viewer(collisionPairCount),
                collisionPairPrefixSum  = make_viewer(collisionPairPrefixSum),
                collisionPairs          = make_viewer(collisionPairs),
                pred = std::forward<Pred>(pred)] __device__(int cell) mutable
               {
                   int size       = objCountInCell(cell);
                   int offset     = objCountInCellPrefixSum(cell);
                   int pairOffset = collisionPairPrefixSum(cell);
                   int index      = 0;
                   for(int i = 0; i < size; ++i)
                   {
                       auto cell0 = cellArrayValueSorted(offset + i);
                       auto oid0  = cell0.oid;
                       auto s0    = spheres(oid0);
                       for(int j = i + 1; j < size; ++j)
                       {
                           auto cell1 = cellArrayValueSorted(offset + j);
                           auto oid1  = cell1.oid;
                           auto s1    = spheres(oid1);
                           //print("test => %d,%d\n", oid0, oid1);
                           if(!Cell::allowIgnore(cell0, cell1)  // cell0, cell1 are created by test the proxy sphere
                              && collide::detect(s0, s1)  // test the bounding spheres to get exact collision result
                              && pred(s0.id, s1.id))  // user predicate
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
}
}  // namespace muda::details