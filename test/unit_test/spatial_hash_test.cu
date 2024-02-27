#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/ext/geo/spatial_hash.h>

using namespace muda;
using namespace muda::spatial_hash;
using namespace Eigen;

void spatial_hash_test()
{
    SparseSpatialHash sh;

    DeviceBuffer<BoundingSphere> spheres;
    DeviceBuffer<CollisionPair>  pairs;
    std::vector<BoundingSphere>  h_spheres;

    for(int i = 0; i < 400; i++)
    {
        float radius = i % 2 == 0 ? 1 : 4.5;
        BoundingSphere s{Vector3f::Ones() * 10 + Vector3f::Random() * 10, radius};
        s.level = i % 2 == 0 ? 0 : 1;
        h_spheres.push_back(s);
    }
    h_spheres.push_back({Vector3f(0, 0, 0), 1});

    spheres = h_spheres;

    for(int level = 0; level < 2; level++)
    {
        sh.detect(level, spheres, pairs);
    }

    std::vector<CollisionPair> pair_data;
    std::vector<CollisionPair> pair_data_ground_truth;
    pairs.copy_to(pair_data);

    for(int i = 0; i < h_spheres.size(); i++)
    {
        for(int j = i + 1; j < h_spheres.size(); j++)
        {
            if(intersect(h_spheres[i], h_spheres[j]))
            {
                pair_data_ground_truth.push_back({i, j});
            }
        }
    }

    std::sort(pair_data.begin(), pair_data.end());
    std::sort(pair_data_ground_truth.begin(), pair_data_ground_truth.end());

    std::vector<CollisionPair> diff;
    std::set_difference(pair_data_ground_truth.begin(),
                        pair_data_ground_truth.end(),
                        pair_data.begin(),
                        pair_data.end(),
                        std::back_inserter(diff));

    CHECK(diff.size() == 0);
    CHECK(pair_data == pair_data_ground_truth);
}

TEST_CASE("spatial_hash_test", "[geo]")
{
    for(int i = 0; i < 100; i++)
        spatial_hash_test();
}
