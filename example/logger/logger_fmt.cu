//#include <catch2/catch.hpp>
//#include <muda/muda.h>
//#include "../example_common.h"
//#include <Eigen/Core>
//#include <fmt/printf.h>
//#include <fmt/args.h>
//using namespace muda;
//
//void logger_fmt()
//{
//    example_desc(R"(This example, we show how to use `fmt` to print info from kernels)");
//
//    Logger logger;
//    Launch(2, 1)
//        .apply([logger = logger.viewer()] __device__() mutable
//               { logger.push_string<true>("fmt string: {}") << 1; })
//        .wait();
//    auto meta = logger.retrieve_meta();
//    fmt::dynamic_format_arg_store<fmt::format_context> store;
//
//    std::string fmt;
//    for(auto m : meta.meta_data())
//    {
//        if(m.type == LoggerBasicType::FmtString)
//            fmt = (const char*)(m.data);
//        else if(m.type == LoggerBasicType::Int)
//            store.push_back(m.as<int>());
//    }
//
//    std::cout << fmt::vformat(fmt, store);
//}
//
//TEST_CASE("logger_fmt", "[logger]")
//{
//    logger_fmt();
//}
