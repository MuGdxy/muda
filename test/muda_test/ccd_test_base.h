#pragma once
#include <muda/container.h>
#include <muda/buffer.h>
#include <iostream>
#include <fstream>


// read test data from csv file
inline void read_ccd_csv(const std::string&            inputFileName,
                           muda::host_vector<Eigen::Vector3f>& X,
                           muda::host_vector<uint32_t>&        res)
{
    // be careful, there are n lines which means there are n/8 queries, but has
    // n results, which means results are duplicated
    std::vector<std::array<double, 3>> vs;
    vs.clear();
    std::ifstream infile;
    infile.open(inputFileName);
    std::array<double, 3> v;
    if(!infile.is_open())
    {
        throw muda::exception("error path" + inputFileName);
    }

    int l = 0;
    while(infile)  // there is input overload classfile
    {
        l++;
        std::string s;
        if(!getline(infile, s))
            break;
        if(s[0] != '#')
        {
            std::istringstream         ss(s);
            std::array<long double, 7> record;  // the first six are one vetex,
                                                // the seventh is the result
            int c = 0;
            while(ss)
            {
                std::string line;
                if(!getline(ss, line, ','))
                    break;
                try
                {
                    record[c] = std::stold(line);
                    c++;
                }
                catch(const std::invalid_argument e)
                {
                    std::cout << "NaN found in file " << inputFileName
                              << " line " << l << std::endl;
                    e.what();
                }
            }
            double x = record[0] / record[1];
            double y = record[2] / record[3];
            double z = record[4] / record[5];
            v[0]     = x;
            v[1]     = y;
            v[2]     = z;

            if(vs.size() % 8 == 0)
                res.push_back(record[6]);
            vs.push_back(v);
        }
    }
    X.resize(vs.size());
    for(int i = 0; i < vs.size(); i++)
    {
        X[i][0] = vs[i][0];
        X[i][1] = vs[i][1];
        X[i][2] = vs[i][2];
    }
    if(!infile.eof())
    {
        std::cerr << "Could not read file " << inputFileName << "\n";
    }
}

inline bool check_allow_false_positive(const muda::host_vector<uint32_t>& ground_thruth,
                                const muda::host_vector<uint32_t>& result)
{
    bool ret = true;
    for(size_t i = 0; i < ground_thruth.size(); ++i)
    {
        if(ground_thruth[i] > result[i])
        {
            ret = false;
            std::cout << "[" << i << "] ground_thruth: " << ground_thruth[i]
                      << " result: " << result[i] << std::endl;
        }
    }
    return ret;
}