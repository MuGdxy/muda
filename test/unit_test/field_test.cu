#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/container.h>
#include <muda/syntax_sugar.h>
#include <Eigen/Core>
using namespace muda;

//namespace muda
//{
//class Field;
//
//class FieldEntryViewerBase : public ViewerBase
//{
//  public:
//    MUDA_GENERIC FieldEntryViewerBase() {}
//
//    MUDA_GENERIC FieldEntryViewerBase(char*                  buffer,
//                                      uint32_t               begin,
//                                      uint32_t               stride,
//                                      uint32_t               elem_alignment,
//                                      uint32_t               elem_count,
//                                      uint32_t               elem_byte_size,
//                                      details::StringPointer name_ptr)
//        : m_buffer(buffer)
//        , m_begin(begin)
//        , m_stride(stride)
//        , m_elem_alignment(elem_alignment)
//        , m_elem_count(elem_count)
//        , m_elem_byte_size(elem_byte_size)
//        , m_name_ptr(name_ptr)
//    {
//    }
//
//  protected:
//    MUDA_GENERIC auto raw_begin() const { return m_buffer + m_begin; }
//
//
//    char*                  m_buffer         = nullptr;
//    uint32_t               m_begin          = ~0;
//    uint32_t               m_stride         = ~0;
//    uint32_t               m_elem_alignment = ~0;
//    uint32_t               m_elem_count     = ~0;
//    uint32_t               m_elem_byte_size = ~0;
//    details::StringPointer m_name_ptr       = {};
//
//  public:
//    MUDA_GENERIC auto elem_byte_size() const { return m_elem_byte_size; }
//    MUDA_GENERIC auto count() const { return m_elem_count; }
//    MUDA_GENERIC auto elem_alignment() const { return m_elem_alignment; }
//    MUDA_GENERIC auto stride() const { return m_stride; }
//    MUDA_GENERIC auto name() const { return m_name_ptr.auto_select(); }
//};
//
//class FieldEntryBase
//{
//  public:
//    auto                 name() const { return std::string{m_name}; }
//    auto                 elem_byte_size() const { return m_elem_byte_size; }
//    auto                 count() const { return m_elem_count; }
//    auto                 elem_alignment() const { return m_elem_alignment; }
//    auto                 stride() const { return m_stride; }
//    FieldEntryViewerBase viewer();
//
//  protected:
//    FieldEntryBase(Field& field, std::string_view name, uint32_t m_elem_byte_size)
//        : m_field(field)
//        , m_name(name)
//        , m_elem_byte_size(m_elem_byte_size)
//    {
//    }
//    ~FieldEntryBase() = default;
//
//    friend class Field;
//    std::string m_name;
//    uint32_t    m_elem_byte_size;
//    Field&      m_field;
//
//    // computed by the field
//    uint32_t               m_begin          = ~0;
//    uint32_t               m_stride         = ~0;
//    uint32_t               m_elem_alignment = ~0;
//    uint32_t               m_elem_count     = ~0;
//    details::StringPointer m_name_ptr;
//};
//template <typename T>
//class FieldEntry;
//
//template <typename T>
//class FieldEntryViewer : public FieldEntryViewerBase
//{
//  public:
//    MUDA_GENERIC T& operator()(int i) const
//    {
//        return *reinterpret_cast<T*>(raw_begin() + m_stride * i);
//    }
//
//  private:
//    friend class FieldEntry<T>;
//    friend class FieldViewer;
//    using FieldEntryViewerBase::FieldEntryViewerBase;
//};
//
//template <typename T>
//class FieldEntry : public FieldEntryBase
//{
//  public:
//    FieldEntry(Field& field, std::string_view name)
//        : FieldEntryBase(field, name, sizeof(T))
//    {
//    }
//
//    void fill(const T& value)
//    {
//        ParallelFor(256)
//            .apply(count(),
//                   [e = viewer(), value = value] __device__(int i) mutable
//                   { e(i) = value; })
//            .wait();
//    }
//
//    void copy_to(BufferView<T> view)
//    {
//        ParallelFor(256)
//            .apply(count(),
//                   [e = viewer(), view = view.viewer()] __device__(int i) mutable
//                   { view(i) = e(i); })
//            .wait();
//    }
//
//    void copy_from(const BufferView<T> view)
//    {
//        ParallelFor(256)
//            .apply(count(),
//                   [e = viewer(), view = view.cviewer()] __device__(int i) mutable
//                   { e(i) = view(i); })
//            .wait();
//    }
//
//    FieldEntryViewer<T> viewer()
//    {
//        return FieldEntryViewer<T>{m_field.m_buffer.data(),
//                                   m_begin,
//                                   m_stride,
//                                   m_elem_alignment,
//                                   m_elem_count,
//                                   m_elem_byte_size,
//                                   m_name_ptr};
//    }
//};
//
//enum class Layout
//{
//    None,
//    AoS,
//    SoA,
//};
//
//class FieldBuildOptions
//{
//  public:
//    Layout   layout    = Layout::SoA;
//    uint32_t alignment = 4;  // bytes
//};
//
//class FieldViewer : public ViewerBase
//{
//    MUDA_VIEWER_COMMON_NAME(FieldViewer);
//
//  private:
//    friend class Field;
//    Dense1D<FieldEntryViewerBase> m_entries;
//    MUDA_GENERIC FieldViewer(const Dense1D<FieldEntryViewerBase> m)
//        : m_entries(m)
//    {
//    }
//
//  public:
//    template <typename T>
//    MUDA_DEVICE FieldEntryViewer<T> entry(const char* name)
//    {
//        auto strcmp = [] MUDA_DEVICE(const char* a, const char* b) -> bool
//        {
//            while(*a && *b && *a == *b)
//            {
//                a++;
//                b++;
//            }
//            return *a == *b;
//        };
//        for(int i = 0; i < m_entries.total_size(); i++)
//        {
//            auto& e = m_entries(i);
//            if(strcmp(e.name(), name))
//            {
//                MUDA_KERNEL_ASSERT(e.elem_byte_size() == size_of<T>(),
//                                   "FieldViewer[%s:%s]: FieldEntry[%s] Type size mismatching, entry type size=%d, your size=%d",
//                                   kernel_name(),
//                                   this->name(),
//                                   name,
//                                   e.elem_byte_size(),
//                                   size_of<T>());
//#if MUDA_CHECK_ON
//                using Acc           = details::ViewerBaseAccessor;
//                Acc::kernel_name(e) = Acc::kernel_name(*this);
//#endif
//                return (FieldEntryViewer<T>&)e;
//            }
//        }
//        MUDA_KERNEL_ERROR_WITH_LOCATION("FieldViewer[%s:%s] FieldEntry[%s] not found",
//                                        kernel_name(),
//                                        this->name(),
//                                        name);
//        return FieldEntryViewer<T>{};
//    }
//};
//
//class Field
//{
//    template <typename T>
//    using U = std::unique_ptr<T>;
//    U<details::HostDeviceStringCache> m_string_cache;
//    FieldBuildOptions                 m_options;
//    std::vector<FieldEntryBase*>      m_entries;
//
//    static constexpr uint32_t round_up(uint32_t value, uint32_t alignment)
//    {
//        return (value + alignment - 1) & ~(alignment - 1);
//    }
//
//    size_t m_resize_base;
//
//  public:
//    DeviceBuffer<char>                         m_buffer;
//    mutable DeviceBuffer<FieldEntryViewerBase> m_entries_buffer;
//
//    template <typename T>
//    FieldEntry<T>& create_entry(std::string_view name)
//    {
//        auto ret = new FieldEntry<T>(*this, name);
//        m_entries.emplace_back(ret);
//        return *ret;
//    }
//
//
//    void build(const FieldBuildOptions& options = {})
//    {
//        m_options     = options;
//        m_resize_base = 0;
//        if(m_options.layout == Layout::SoA)
//        {
//            build_soa(options);
//        }
//        else
//        {
//            // build_aos(options);
//        }
//    }
//
//    void build_soa(const FieldBuildOptions& options = {})
//    {
//        size_t string_size = 0;
//        for(auto entry : m_entries)
//        {
//            string_size += entry->m_name.size() + 1;
//            // unknown yet, we dont know the size of the entry at this point
//            entry->m_begin = ~0;
//            entry->m_stride = round_up(entry->m_elem_byte_size, options.alignment);
//            entry->m_elem_alignment = options.alignment;
//            m_resize_base += entry->m_stride;
//        }
//        m_string_cache = std::make_unique<details::HostDeviceStringCache>(string_size);
//        for(auto entry : m_entries)
//        {
//            entry->m_name_ptr = (*m_string_cache)[entry->m_name];
//        }
//    }
//
//    auto size() const { return m_buffer.size() / m_resize_base; }
//
//    void resize_set_soa()
//    {
//        uint32_t s      = size();
//        uint32_t offset = 0;
//        for(auto entry : m_entries)
//        {
//            entry->m_begin      = offset;
//            entry->m_elem_count = s;
//            offset += s * entry->m_stride;
//        }
//    }
//
//    void resize(size_t num_elements)
//    {
//        m_buffer.resize(num_elements * m_resize_base);
//        if(m_options.layout == Layout::SoA)
//        {
//            resize_set_soa();
//        }
//        else
//        {
//            // resize_set_aos();
//        }
//    }
//
//    void upload_entries() const
//    {
//        m_entries_buffer.resize(m_entries.size());
//        std::vector<FieldEntryViewerBase> entries(m_entries.size());
//        std::transform(m_entries.begin(),
//                       m_entries.end(),
//                       entries.begin(),
//                       [](auto entry) { return entry->viewer(); });
//        m_entries_buffer = entries;
//    }
//
//    FieldViewer viewer() const
//    {
//        if(!m_entries_buffer.size())
//            upload_entries();
//        return FieldViewer{m_entries_buffer.viewer()};
//    }
//
//    ~Field()
//    {
//        for(auto entry : m_entries)
//        {
//            delete entry;
//        }
//    }
//};
//
//void field_test()
//{
//    using Vector3 = Eigen::Vector3f;
//
//    using namespace Eigen;
//
//    Field field;
//    auto& x = field.create_entry<float>("x");
//    auto& y = field.create_entry<float>("y");
//    auto& z = field.create_entry<float>("z");
//    field.build();
//
//    field.resize(4);
//
//    x.fill(1.0f);
//    y.fill(2.0f);
//    z.fill(3.0f);
//    auto N = field.size();
//    ParallelFor(256)
//        .apply(N,
//               [x = x.viewer(), y = y.viewer(), z = z.viewer()] $(int i)
//               {
//                   print("%s(%d)=%f %s(%d)=%f %s(%d)=%f\n",
//                         x.name(),
//                         i,
//                         x(i),
//                         y.name(),
//                         i,
//                         y(i),
//                         z.name(),
//                         i,
//                         z(i));
//               })
//        .wait();
//
//    DeviceBuffer<float> result(N);
//    x.copy_to(result);
//    ParallelFor(256)
//        .apply(N,
//               [result = result.viewer()] $(int i)
//               { print("result(%d)=%f\n", i, result(i)); })
//        .wait();
//
//    y.copy_from(result);
//    ParallelFor(256)
//        .apply(N, [y = y.viewer()] $(int i) { print("y(%d)=%f\n", i, y(i)); })
//        .wait();
//
//    std::cout << "Using Field:\n";
//
//    auto v = field.viewer();
//
//
//    ParallelFor(256)
//        .kernel_name("field")
//        .apply(N,
//               [field = field.viewer()] $(int i)
//               {
//                   auto x = field.entry<float>("x");
//                   auto y = field.entry<float>("y");
//                   auto z = field.entry<float>("z");
//
//                   auto vec_stride = &y(0) - &x(0);
//                   print("vec_stride=%lld\n", vec_stride);
//                   Map<Vector3f, 0, InnerStride<Dynamic>> v(&x(i), InnerStride<Dynamic>(4));
//                   v *= 2.0f;
//
//                   print("%s(%d)=%f %s(%d)=%f %s(%d)=%f\n",
//                         x.name(),
//                         i,
//                         x(i),
//                         y.name(),
//                         i,
//                         y(i),
//                         z.name(),
//                         i,
//                         z(i));
//               })
//        .wait();
//}

//TEST_CASE("field_test", "[field]")
//{
//    field_test();
//}

//MUDA_INLINE FieldEntryViewerBase FieldEntryBase::viewer()
//{
//    return FieldEntryViewerBase{
//        m_field.m_buffer.data(), m_begin, m_stride, m_elem_alignment, m_elem_count, m_elem_byte_size, m_name_ptr};
//}

#include <muda/field/field.h>
#include <muda/field/field_builder.h>
#include <muda/ext/eigen.h>

using namespace Eigen;
using namespace muda::eigen;

void field_test1(FieldEntryLayout layout)
{
    Field field;
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field
    auto  builder = particle.builder(FieldEntryLayout::SoA);
    auto& m       = builder.entry("mass").scalar<float>();
    auto& pos     = builder.entry("position").vector3<float>();
    auto& vel     = builder.entry("velocity").vector3<float>();
    auto& force   = builder.entry("force").vector3<float>();
    auto& I       = builder.entry("inertia").matrix3x3<float>();
    builder.build();

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    ParallelFor(256)
        .apply(N,
               [m   = make_viewer(m),
                pos = make_viewer(pos),
                vel = make_viewer(vel),
                f   = make_viewer(force),
                I   = make_viewer(I)] $(int i)
               {
                   m(i)   = 1.0f;
                   pos(i) = Vector3f::Zero();
                   vel(i) = Vector3f::Zero();
                   f(i)   = Vector3f{0.0f, -9.8f, 0.0f};

                   I(i, 1, 0) = 1;
                   I(i) += Matrix3f::Ones();
                   f.name();
                   auto x = pos(i);
                   print("position = %f %f %f\n", x.x(), x.y(), x.z());
               })
        .wait();

    particle.resize(N * 13);

    ParallelFor(256)
        .apply(N,
               [m   = m.viewer(),
                pos = make_viewer(pos),
                vel = make_viewer(vel),
                f   = make_cviewer(force),
                I   = make_cviewer(I),
                dt] $(int i)
               {
                   auto     x = pos(i);
                   auto     v = vel(i);
                   Vector3f a = f(i) / m(i);

                   v = v + a * dt;
                   x = x + v * dt;
                   print("position = %f %f %f\n", x.x(), x.y(), x.z());
                   print("innerta diag = %f %f %f\n", I(i)(0, 0), I(i)(1, 1), I(i)(2, 2));
                   print("innerta(1,0) = %f\n", I(i)(1, 0));
               })
        .wait();
}

void field_test2(FieldEntryLayout layout)
{
    Field field;
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field
    auto builder = particle.builder(layout);
    // auto  builder = particle.builder<FieldEntryLayout::SoA>();
    auto& m = builder.entry("mass").scalar<float>();
    auto& I = builder.entry("inertia").matrix3x3<float>();
    builder.build();

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    Logger logger;
    ParallelFor(256)
        .apply(N,
               [m = m.viewer(), I = make_viewer(I), logger = logger.viewer()] $(int i)
               {
                   m(i) = 1.0f;
                   //pos(i) = Vector3f::Ones();
                   I(i, 0, 0) = 1;
                   I(i, 1, 0) = 2;
                   I(i, 2, 0) = 3;
                   I(i, 0, 1) = 4;
                   I(i, 1, 1) = 5;
                   I(i, 2, 1) = 6;
                   I(i, 0, 2) = 7;
                   I(i, 1, 2) = 8;
                   I(i, 2, 2) = 9;
                   logger << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "I=\n"
                          << I(i) << "\n";
               })
        .wait();
    logger.retrieve();

    particle.resize(N * 2);

    ParallelFor(256)
        .apply(N,
               [m = m.viewer(), I = make_viewer(I), logger = logger.viewer()] $(int i)
               {
                   logger << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "I=\n"
                          << I(i) << "\n";
               })
        .wait();
}

TEST_CASE("field_test", "[field]")
{
    using Layout = FieldEntryLayout;
    field_test1(Layout::AoS);
    field_test1(Layout::SoA);
    field_test1(Layout::AoSoA);
    field_test2(Layout::AoS);
    field_test2(Layout::SoA);
    field_test2(Layout::AoSoA);
}