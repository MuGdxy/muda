//#include <catch2/catch.hpp>
//#include <muda/muda.h>
//#include <example_common.h>
//#include <Eigen/Core>
//#include <fmt/printf.h>
//#include <fmt/args.h>
//#include <fmt/core.h>
//using namespace muda;
//
//struct point
//{
//    double x, y;
//
//    __host__ static void _fmt_arg(void* fmtter, const void* obj)
//    {
//        auto& fmt =
//            *reinterpret_cast<fmt::dynamic_format_arg_store<fmt::format_context>*>(fmtter);
//        auto& p = *reinterpret_cast<const point*>(obj);
//        fmt.push_back(p);
//    }
//};
//
//template <>
//struct fmt::formatter<point>
//{
//    // Presentation format: 'f' - fixed, 'e' - exponential.
//    char presentation = 'f';
//
//    // Parses format specifications of the form ['f' | 'e'].
//    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator
//    {
//        // [ctx.begin(), ctx.end()) is a character range that contains a part of
//        // the format string starting from the format specifications to be parsed,
//        // e.g. in
//        //
//        //   fmt::format("{:f} - point of interest", point{1, 2});
//        //
//        // the range will contain "f} - point of interest". The formatter should
//        // parse specifiers until '}' or the end of the range. In this example
//        // the formatter should parse the 'f' specifier and return an iterator
//        // pointing to '}'.
//
//        // Please also note that this character range may be empty, in case of
//        // the "{}" format string, so therefore you should check ctx.begin()
//        // for equality with ctx.end().
//
//        // Parse the presentation format and store it in the formatter:
//        auto it = ctx.begin(), end = ctx.end();
//        if(it != end && (*it == 'f' || *it == 'e'))
//            presentation = *it++;
//
//        // Check if reached the end of the range:
//        if(it != end && *it != '}')
//            throw_format_error("invalid format");
//
//        // Return an iterator past the end of the parsed range:
//        return it;
//    }
//
//    // Formats the point p using the parsed format specification (presentation)
//    // stored in this formatter.
//    auto format(const point& p, format_context& ctx) const -> format_context::iterator
//    {
//        // ctx.out() is an output iterator to write to.
//        return presentation == 'f' ?
//                   fmt::format_to(ctx.out(), "({:.1f}, {:.1f})", p.x, p.y) :
//                   fmt::format_to(ctx.out(), "({:.1e}, {:.1e})", p.x, p.y);
//    }
//};
//
//
//void logger_fmt()
//{
//    example_desc(R"(This example, we show how to use `fmt` to print info from kernels)");
//    point;
//    auto   fmt_arg = point::_fmt_arg;
//    Logger logger;
//    Launch(1, 1)
//        .apply(
//            [=, logger = logger.viewer()] __device__() mutable
//            {
//                point p{1.0, 2.0};
//                print("fmt_arg=%d", (int)fmt_arg);
//                logger.push_string<true>("fmt string: {}").push_fmt_arg(p, fmt_arg);
//            })
//        .wait();
//    auto meta = logger.retrieve_meta();
//    fmt::dynamic_format_arg_store<fmt::format_context> store;
//
//    std::string fmt;
//    for(auto m : meta.meta_data())
//    {
//        if(m.type == LoggerBasicType::FmtString)
//            fmt = (const char*)(m.data);
//        else if(m.type == LoggerBasicType::Object)
//        {
//            m.fmt_arg(&store, m.data);
//        }
//    }
//
//    std::cout << fmt::vformat(fmt, store);
//}
//
//TEST_CASE("logger_fmt", "[logger]")
//{
//    logger_fmt();
//}
