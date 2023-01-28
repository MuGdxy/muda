#include <catch2/catch.hpp>
#include <algorithm>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/buffer.h>
#include <muda/pba/collision/spatial_hash.h>

using namespace muda;

// if <input_file_name> is empty, we generate a random input
// otherwise, we read the input from the file
// input format:
// ox, oy, oz, r, id
std::string input_file_name = "";
// if the result of the broad phase collision detection has incoherence with the ground truth
// we output the difference to a file with name <output_file_name> for further visualization and checking
std::string output_file_name = "diff.csv";

template <typename Hash>
void spatial_hash_test(host_vector<CollisionPair>& res, host_vector<CollisionPair>& gt)
{
    host_vector<sphere> h_spheres;
    if(input_file_name.empty())  // or generate random test set
    {
        h_spheres.resize(6400);
        std::generate(h_spheres.begin(),
                      h_spheres.end(),
                      [id = uint32_t(0)]() mutable
                      {
                          return sphere(
                              Eigen::Vector3f(0.2 + rand() / (float)RAND_MAX * 10,
                                              0.2 + rand() / (float)RAND_MAX * 10,
                                              0.2 + rand() / (float)RAND_MAX * 10),
                              0.2,
                              id++);
                      });
    }
    else  // read from input file
    {
        std::ifstream ifs;
        ifs.open(input_file_name);
        int i = 0;
        while(true)
        {
            sphere s;
            s.from_csv(ifs);
            i++;
            if(!ifs)
                break;
            h_spheres.push_back(s);
        }
        std::cout << "we read " << h_spheres.size()
                  << " spheres from: " << input_file_name << std::endl;
        ifs.close();
    }


    stream s;
    auto   spheres = to_device(h_spheres);
    launch::wait_device();
    SpatialPartitionField<Hash>  field;
    device_buffer<CollisionPair> d_res;

    on(s)
        .next<SpatialPartitionLauncher<Hash>>(field)  // setup config
        .configSpatialHash(Eigen::Vector3f(0, 0, 0),  // give the left-bottom corner of the domain
                           1.0f)  // set cell size manually which will disable automatic cell size calculation
        .applyCreateCollisionPairs(make_viewer(spheres), d_res);

    d_res.copy_to(res);      // this copy is also async
    launch::wait_stream(s);  // wait for the copy to finish

    // detect collision in these 1000 spheres brutely
    for(int i = 0; i < h_spheres.size(); i++)
        for(int j = i + 1; j < h_spheres.size(); j++)
            if(collide::detect(h_spheres[i], h_spheres[j]))
                gt.push_back(CollisionPair(h_spheres[i].id, h_spheres[j].id));

    std::sort(gt.begin(), gt.end());
    std::sort(res.begin(), res.end());
    if(gt.size() != res.size())
        std::cout << "incoherence: gt-size =" << gt.size()
                  << " res-size =" << res.size() << std::endl;
    std::vector<CollisionPair> difference;
    std::set_difference(gt.begin(), gt.end(), res.begin(), res.end(), std::back_inserter(difference));

    if(difference.size())
    {
        std::ofstream o;
        o.open(output_file_name);
        for(auto&& p : difference)
        {
            auto s0 = h_spheres[p.id[0]];
            s0.to_csv(o) << std::endl;
            auto s1 = h_spheres[p.id[1]];
            s1.to_csv(o) << std::endl;
            std::cout << "(" << p << ")" << std::endl;
        }
        o.close();
    }

    {
        std::ofstream o;
        o.open("input.csv");
        for(auto&& c : h_spheres)
        {
            c.to_csv(o) << std::endl;
        }
        o.close();
    }
}

TEST_CASE("spatial_hash", "[collide]")
{
    SECTION("shift_hash")
    {
        host_vector<CollisionPair> res, gt;
        spatial_hash_test<shift_hash<20, 10, 0>>(res, gt);
        REQUIRE(res == gt);
    }
    SECTION("morton")
    {
        host_vector<CollisionPair> res, gt;
        spatial_hash_test<morton>(res, gt);
        REQUIRE(res == gt);
    }
}