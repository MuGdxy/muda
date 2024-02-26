namespace muda::spatial_hash::details
{
//using Hash = Morton<int32_t>;
//using Pred = DefaultPredication;

template <typename Hash>
inline void SparseSpatialHashImpl<Hash>::beginSetupHashTable()
{
    beginFillHashCells();
    beginSortHashCells();
    beginCountObjectPerCell();
}

template <typename Hash>
template <typename Pred>
inline void SparseSpatialHashImpl<Hash>::beginCreateCollisionPairs(
    CBufferView<BoundingSphere>  boundingSphereList,
    DeviceBuffer<CollisionPair>& collisionPairs,
    Pred&&                       pred)
{
    spheres = boundingSphereList;

    beginCalculateCellSizeAndCoordMin();
    beginSetupHashTable();
    waitAndCreateTempData();

    beginCountCollisionPairs(std::forward<Pred>(pred));
    waitAndAllocCollisionPairList(collisionPairs);
    beginSetupCollisionPairList(collisionPairs, std::forward<Pred>(pred));
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::beginCalculateCellSizeAndCoordMin()
{
    auto count = spheres.size();

    BufferLaunch(m_stream)
        .clear(allRadius)
        .resize(allRadius, count)
        .clear(allCoords)
        .resize(allCoords, count);

    ParallelFor(0, m_stream)  //
        .apply(count,
               [spheres   = spheres.cviewer().name("spheres"),
                allRadius = allRadius.viewer().name("all_radius"),
                allCoords = allCoords.viewer().name("all_coords"),
                level     = this->level] __device__(int i) mutable
               {
                   const auto& s = spheres(i);
                   // allRadius(i)  = s.level <= level ? s.r : 0.0f;
                   allRadius(i) = s.r;
                   allCoords(i) = s.o;
               });

    DeviceReduce().Max(allRadius.data(), maxRadius.data(), count);

    constexpr auto float_max = std::numeric_limits<float>::max();

    DeviceReduce().Reduce(
        allCoords.data(),
        minCoord.data(),
        count,
        [] __host__ __device__(const Vector3& a, const Vector3& b) -> Vector3
        { return a.cwiseMin(b); },
        Vector3{float_max, float_max, float_max});

    // Scaling the bounding sphere of each object by sqrt(2) [we will use proxy spheres]
    // and ensuring that the grid cell is at least 1.5 times
    // as large as the scaled bounding sphere of the largest object.
    //https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

    auto scaledMaxRadius          = maxRadius * 2 * 1.5 * 1.5;
    h_spatialHashConfig.coord_min = minCoord;
    // shift the coord_min by the scaledMaxRadius, which is much safer than the original maxRadius
    h_spatialHashConfig.coord_min -= scaledMaxRadius * Vector3::Ones();
    h_spatialHashConfig.cell_size = scaledMaxRadius;

    // upload
    spatialHashConfig = h_spatialHashConfig;
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::beginFillHashCells()
{
    using namespace muda;

    int size  = spheres.size();
    int count = 8 * size + 1;  //

    BufferLaunch(m_stream)
        //.clear(cellArrayValue)
        .resize(cellArrayValue, count)
        //.clear(cellArrayKey)
        .resize(cellArrayKey, count)
        //.clear(cellArrayValueSorted)
        .resize(cellArrayValueSorted, count)
        //.clear(cellArrayKeySorted)
        .resize(cellArrayKeySorted, count)
        //.clear(uniqueKey)
        .resize(uniqueKey, count)
        //.clear(objCountInCell)
        .resize(objCountInCell, count)
        //.clear(objCountInCellPrefixSum)
        .resize(objCountInCellPrefixSum, count)
        //.clear(collisionPairCount)
        .resize(collisionPairCount, count)
        //.clear(collisionPairPrefixSum)
        .resize(collisionPairPrefixSum, count);

    ParallelFor(0, m_stream)  //
        .apply(spheres.size(),
               [spheres           = spheres.cviewer(),
                spatialHashConfig = spatialHashConfig.viewer(),
                cellArrayValue = make_dense_2d(cellArrayValue.data(), size, 8),
                cellArrayKey   = make_dense_2d(cellArrayKey.data(), size, 8),
                level          = this->level] __device__(int i) mutable
               {
                   BoundingSphere s  = spheres(i);
                   auto&          sh = *spatialHashConfig;

                   auto proxySphere = s;
                   // https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
                   // Scaling the bounding sphere of each object by sqrt(2), here we take 1.5(>1.414)
                   proxySphere.r *= 1.5;

                   auto     o        = s.o;
                   Vector3u ijk      = sh.cell(o);
                   auto     hash     = sh.hash_cell(ijk);
                   auto     cellSize = sh.cell_size;
                   // print("beginFillHashCells cellSize=%f\n", cellSize);
                   auto objectId = i;


                   Cell homeCell(hash, objectId);

                   // ...[i*8+0][i*8+1][i*8+2][i*8+3][i*8+4][i*8+5][i*8+6][i*8+7]...
                   // ...[hcell][pcell][pcell] ...   [  x  ][  x  ][  x  ][  x  ]...


                   // fill the cell that contains the center of the current sphere
                   homeCell.set_as_home(ijk);
                   homeCell.ijk = ijk;
                   Vector3 xyz  = sh.cell_center_coord(ijk);

                   //find the cloest 7 neighbor cells
                   Vector3i dxyz;
#pragma unroll
                   for(int i = 0; i < 3; ++i)
                       dxyz(i) = o(i) > xyz(i) ? 1 : -1;

                   auto cal_cell = [&](const Vector3i& dxyz) -> Vector3u

                   {
                       Vector3i res = (ijk.cast<I32>() + dxyz);
                       return res.cast<U32>();
                   };

                   Vector3u cells[7] = {cal_cell(Vector3i(dxyz.x(), 0, 0)),
                                        cal_cell(Vector3i(0, dxyz.y(), 0)),
                                        cal_cell(Vector3i(0, 0, dxyz.z())),
                                        cal_cell(Vector3i(0, dxyz.y(), dxyz.z())),
                                        cal_cell(Vector3i(dxyz.x(), 0, dxyz.z())),
                                        cal_cell(Vector3i(dxyz.x(), dxyz.y(), 0)),
                                        cal_cell(dxyz)};

                   // the cell size (3d)
                   Vector3 size(cellSize, cellSize, cellSize);

                   int idx = 1;  //goes from 1 -> 7. idx = 0 is for the homeCell
#pragma unroll
                   for(int i = 0; i < 7; ++i)
                   {
                       Vector3 min = sh.coord(cells[i]);
                       Vector3 max = min + size;
                       AABB    aabb(min, max);

                       // use proxySphere to test
                       // whether the current sphere overlaps the neighbor cell
                       if(intersect(proxySphere, aabb))
                       {
                           homeCell.set_overlap(cells[i]);
                           auto hash = sh.hash_cell(cells[i]);
                           Cell phantom(hash, objectId);
                           phantom.set_as_phantom(ijk, cells[i]);
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

    Launch(1, 1, 0, m_stream)  //
        .apply(
            [cellArrayValue = cellArrayValue.viewer(),
             cellArrayKey   = cellArrayKey.viewer()] __device__() mutable
            {
                cellArrayKey(cellArrayKey.total_size() - 1)     = -1;
                cellArrayValue(cellArrayValue.total_size() - 1) = Cell(-1, -1);
            });
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::beginSortHashCells()
{
    DeviceRadixSort(m_stream).SortPairs((uint32_t*)cellArrayKey.data(),  //in
                                        (uint32_t*)cellArrayKeySorted.data(),  //out
                                        cellArrayValue.data(),        //in
                                        cellArrayValueSorted.data(),  //out
                                        cellArrayValue.size());
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::beginCountObjectPerCell()
{
    auto count = uniqueKey.size();
    DeviceRunLengthEncode(m_stream).Encode(cellArrayKeySorted.data(),  // in
                                           uniqueKey.data(),           // out
                                           objCountInCell.data(),      // out
                                           uniqueKeyCount.data(),      // out
                                           count);
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::waitAndCreateTempData()
{
    on(m_stream).wait();

    // wait for uniqueKeyCount
    // we need use uniqueKeyCount to resize arrays

    int h_uniqueKeyCount = uniqueKeyCount;

    BufferLaunch(m_stream)
        .resize(uniqueKey, h_uniqueKeyCount)
        .resize(objCountInCell, h_uniqueKeyCount)
        .resize(objCountInCellPrefixSum, h_uniqueKeyCount)
        .resize(collisionPairCount, h_uniqueKeyCount)
        .resize(collisionPairPrefixSum, h_uniqueKeyCount);

    // validCellCount is always 1 less than the uniqueKeyCount,
    // the last one is typically the cell-object-pair {cid = -1, oid = -1}
    validCellCount = h_uniqueKeyCount - 1;
}

template <typename Hash>
template <typename Pred>
void SparseSpatialHashImpl<Hash>::beginCountCollisionPairs(Pred&& pred)
{
    using CallableType = raw_type_t<Pred>;

    using namespace muda;

    // we still prefix sum the uniqueKeyCount cell-object-pair
    // because we need to know the total number of collision pairs
    // which is at the last element of the prefix sum array
    DeviceScan(m_stream).ExclusiveSum(objCountInCell.data(),           // in
                                      objCountInCellPrefixSum.data(),  // out
                                      validCellCount + 1);

    muda::ParallelFor(0, m_stream)  //
        .apply(validCellCount,
               [spheres                 = spheres.viewer(),
                objCountInCell          = objCountInCell.viewer(),
                objCountInCellPrefixSum = objCountInCellPrefixSum.viewer(),
                cellArrayValueSorted    = cellArrayValueSorted.viewer(),
                collisionPairCount      = collisionPairCount.viewer(),
                pred                    = std::forward<Pred>(pred),
                level = this->level] __device__(int cell) mutable
               {
                   int size      = objCountInCell(cell);
                   int offset    = objCountInCellPrefixSum(cell);
                   int pairCount = 0;
                   // print("cell %d has %d objects\n", cell, size);
                   for(int i = 0; i < size; ++i)
                   {
                       auto cell0 = cellArrayValueSorted(offset + i);
                       auto oid0  = cell0.oid;
                       auto s0    = spheres(oid0);
                       if(s0.level < level)
                           continue;
                       for(int j = i + 1; j < size; ++j)
                       {
                           auto cell1 = cellArrayValueSorted(offset + j);
                           auto oid1  = cell1.oid;
                           auto s1    = spheres(oid1);
                           // print("test => %d,%d\n", oid0, oid1);
                           if(!Cell::allow_ignore(cell0, cell1) && intersect(s0, s1)
                              && pred(oid0, oid1))  // user predicate
                           {
                               //print("pair");
                               ++pairCount;
                           }
                       }
                   }
                   collisionPairCount(cell) = pairCount;
               })
        .wait();

    int keycount = uniqueKeyCount;
    if(keycount)
    {
        DeviceScan(m_stream).ExclusiveSum(
            collisionPairCount.data(), collisionPairPrefixSum.data(), keycount);
        auto lastOffset = validCellCount;

        BufferLaunch(m_stream)  //
            .copy(&sum, collisionPairPrefixSum.view(lastOffset));
    }
    else
        sum = 0;
}

template <typename Hash>
void SparseSpatialHashImpl<Hash>::waitAndAllocCollisionPairList(DeviceBuffer<CollisionPair>& collisionPairs)
{
    on(m_stream).wait();

    int totalCollisionPairCount = sum;
    pairListOffset              = collisionPairs.size();
    collisionPairs.resize(collisionPairs.size() + totalCollisionPairCount);
}

template <typename Hash>
template <typename Pred>
void SparseSpatialHashImpl<Hash>::beginSetupCollisionPairList(DeviceBuffer<CollisionPair>& collisionPairs,
                                                              Pred&& pred)
{
    using namespace muda;

    muda::ParallelFor(0, m_stream)
        .apply(validCellCount,
               [spheres                 = spheres.viewer(),
                objCountInCell          = objCountInCell.viewer(),
                objCountInCellPrefixSum = objCountInCellPrefixSum.viewer(),
                cellArrayValueSorted    = cellArrayValueSorted.viewer(),
                collisionPairCount      = collisionPairCount.viewer(),
                collisionPairPrefixSum  = collisionPairPrefixSum.viewer(),
                collisionPairs          = collisionPairs.viewer(),
                pairListOffset          = pairListOffset,
                pred                    = std::forward<Pred>(pred),
                level = this->level] __device__(int cell) mutable
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
                           if(s0.level < level)
                               continue;
                           if(!Cell::allow_ignore(cell0, cell1)  // cell0, cell1 are created by test the proxy sphere
                              && intersect(s0, s1)  // test the bounding spheres to get exact collision result
                              && pred(oid0, oid1))  // user predicate
                           {
                               CollisionPair p;
                               p.id[0] = oid0;
                               p.id[1] = oid1;
                               collisionPairs(pairListOffset + pairOffset + index) = p;
                               ++index;
                           }
                       }
                   }
               });
}
}  // namespace muda::spatial_hash::details