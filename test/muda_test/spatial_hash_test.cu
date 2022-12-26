#include <catch2/catch.hpp>
#include <algorithm>
#include <muda/muda.h>
#include <muda/PBA/collision/spatial_hash.h>

using namespace muda;

void spatial_hash_test()
{
    // generate 1000 spheres with random positions in (0,0,0),(10,10,10) and radii in (0.1,0.2)
    host_vector<sphere> h_spheres(1000);
    std::generate(h_spheres.begin(),
                  h_spheres.end(),
                  [id = uint32_t(0)]() mutable
                  {
                      return sphere(Eigen::Vector3f(rand() / (float)RAND_MAX * 10,
                                                    rand() / (float)RAND_MAX * 10,
                                                    rand() / (float)RAND_MAX * 10),
                                    rand() / (float)RAND_MAX * 0.1 + 0.1,
                                    id++);
                  });
    auto spheres = to_device(h_spheres);
	
    SpatialPartition().prepare(make_viewer(spheres));

	
    // detect collision in these 1000 spheres brutely
    host_vector<CollisionPair> collision_pairs;
    for(int i = 0; i < h_spheres.size(); i++)
        for(int j = i + 1; j < h_spheres.size(); j++)
            if(collide::detect(h_spheres[i], h_spheres[j]))
                collision_pairs.push_back({h_spheres[i].id, h_spheres[j].id});
}

TEST_CASE("spatial_hash", "[collide]") 
{
    spatial_hash_test();
}