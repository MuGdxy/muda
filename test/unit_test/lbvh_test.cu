#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>

#include <muda/ext/geo/lbvh.h>
#include <random>
#include <vector>
#include <thrust/random.h>

using namespace muda;

struct AABBGetter
{
    __device__ __host__ lbvh::AABB<float> operator()(const float4 f) const noexcept
    {
        lbvh::AABB<float> retval;
        retval.upper = f;
        retval.lower = f;
        return retval;
    }
};

struct DistanceCalculator
{
    __device__ __host__ float operator()(const float4 point, const float4 object) const noexcept
    {
        return (point.x - object.x) * (point.x - object.x)
               + (point.y - object.y) * (point.y - object.y)
               + (point.z - object.z) * (point.z - object.z);
    }
};

void lbvh_test()
{
    constexpr std::size_t N = 10;
    std::vector<float4>   ps(N);

    std::mt19937                          mt(123456789);
    std::uniform_real_distribution<float> uni(0.0, 1.0);

    for(auto& p : ps)
    {
        p.x = uni(mt);
        p.y = uni(mt);
        p.z = uni(mt);
    }

    lbvh::BVH<float, float4, AABBGetter> bvh;
    bvh.objects() = ps;
    bvh.build();

    std::cout << "testing query_device:overlap ...\n";
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<std::size_t>(0),
        thrust::make_counting_iterator<std::size_t>(N),
        [bvh = bvh.viewer().name("bvh")] __device__(std::size_t idx) mutable
        {
            unsigned int buffer[10];
            const auto   self = bvh.object(idx);
            const float  dr   = 0.1f;
            for(std::size_t i = 1; i < 10; ++i)
            {
                uint32_t current_idx = 0;
                for(unsigned int j = 0; j < 10; ++j)
                {
                    buffer[j] = 0xFFFFFFFF;
                }


                const float       r = dr * i;
                lbvh::AABB<float> query_box;
                query_box.lower = make_float4(self.x - r, self.y - r, self.z - r, 0);
                query_box.upper = make_float4(self.x + r, self.y + r, self.z + r, 0);
                const auto num_found = bvh.query(lbvh::overlaps(query_box),
                                                 [&] __device__(uint32_t obj_idx) mutable
                                                 {
                                                     if(current_idx < 10)
                                                     {
                                                         buffer[current_idx++] = obj_idx;
                                                     }
                                                 });

                for(unsigned int j = 0; j < 10; ++j)
                {
                    const auto jdx = buffer[j];
                    if(j >= num_found)
                    {
                        assert(jdx == 0xFFFFFFFF);
                        continue;
                    }
                    else
                    {
                        assert(jdx != 0xFFFFFFFF);
                        assert(jdx < bvh.num_objects());
                    }
                    const auto other = bvh.object(jdx);
                    assert(fabsf(self.x - other.x) < r);  // check coordinates
                    assert(fabsf(self.y - other.y) < r);  // are in the range
                    assert(fabsf(self.z - other.z) < r);  // of query box
                }
            }
            return;
        });

    std::cout << "testing query_device:nearest_neighbor ...\n";
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<unsigned int>(0),
                     thrust::make_counting_iterator<unsigned int>(N),
                     [bvh = bvh.viewer()] __device__(const unsigned int idx) mutable
                     {
                         const auto self = bvh.object(idx);
                         const auto nest =
                             bvh.query(lbvh::nearest(self), DistanceCalculator());
                         assert(nest.first != 0xFFFFFFFF);
                         const auto other = bvh.object(nest.first);
                         // of course, the nearest object is itself.
                         assert(nest.second == 0.0f);
                         assert(self.x == other.x);
                         assert(self.y == other.y);
                         assert(self.z == other.z);
                         return;
                     });

    thrust::device_vector<float4> random_points(N);
    thrust::transform(thrust::make_counting_iterator<unsigned int>(0),
                      thrust::make_counting_iterator<unsigned int>(N),
                      random_points.begin(),
                      [] __device__ __host__(const unsigned int idx)
                      {
                          thrust::default_random_engine rand;
                          thrust::uniform_real_distribution<float> uni(0.0f, 1.0f);
                          rand.discard(idx);
                          const float x = uni(rand);
                          const float y = uni(rand);
                          const float z = uni(rand);
                          return make_float4(x, y, z, 0);
                      });

    thrust::for_each(random_points.begin(),
                     random_points.end(),
                     [bvh = bvh.viewer()] __device__(const float4 pos) mutable
                     {
                         const auto calc = DistanceCalculator();
                         const auto nest = bvh.query(lbvh::nearest(pos), calc);
                         assert(nest.first != 0xFFFFFFFF);

                         for(unsigned int i = 0; i < bvh.num_objects(); ++i)
                         {
                             const auto dist = calc(bvh.object(i), pos);
                             if(i == nest.first)
                             {
                                 assert(dist == nest.second);
                             }
                             else
                             {
                                 assert(dist >= nest.second);
                             }
                         }
                         return;
                     });

    std::cout << "testing query_host:overlap ...\n";
    {
        for(std::size_t i = 0; i < 10; ++i)
        {
            const auto  self = bvh.host_objects()[i];
            const float dr   = 0.1f;
            for(unsigned int cnt = 1; cnt < 10; ++cnt)
            {
                const float       r = dr * cnt;
                lbvh::AABB<float> query_box;
                query_box.lower = make_float4(self.x - r, self.y - r, self.z - r, 0);
                query_box.upper = make_float4(self.x + r, self.y + r, self.z + r, 0);

                std::vector<std::size_t> buffer;
                const auto               num_found =
                    lbvh::query(bvh, lbvh::overlaps(query_box), std::back_inserter(buffer));

                for(unsigned int jdx : buffer)
                {
                    assert(jdx < bvh.host_objects().size());

                    const auto other = bvh.host_objects()[jdx];
                    assert(fabsf(self.x - other.x) < r);  // check coordinates
                    assert(fabsf(self.y - other.y) < r);  // are in the range
                    assert(fabsf(self.z - other.z) < r);  // of query box
                }
            }
        }
    }
}

TEST_CASE("lbvh_test", "[geo]")
{
    lbvh_test();
}
