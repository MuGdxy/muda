viewer_template='''#include <muda/viewer/viewer_base.h>

namespace muda
{
template <typename T, int N>
class $NAME$ViewerBase : public muda::ViewerBase
{
  protected:
    // data
    T* m_data;

  public:
    MUDA_GENERIC $NAME$ViewerBase() = default;
    MUDA_GENERIC $NAME$ViewerBase(T* data)
        : m_data(data)
    {
        MUDA_KERNEL_ASSERT(data != nullptr,
                           "$NAME$Viewer [%s:%s]: data is nullptr",
                           name(),
                           kernel_name());
    }

    MUDA_GENERIC $NAME$ViewerBase<T, N> subview(int offset, int count) const
    {
        return $NAME$ViewerBase{};
    }
};

template <typename T, int N>
class C$NAME$Viewer : public $NAME$ViewerBase<T, N>
{
    using Base = $NAME$ViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME(C$NAME$Viewer);

  public:
    MUDA_GENERIC C$NAME$Viewer(const T* data)
        : Base(const_cast<T*>(data))
    {
    }

    MUDA_GENERIC C$NAME$Viewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC C$NAME$Viewer<T, N> subview(int offset, int count) const
    {
        return C$NAME$Viewer{Base::subview(offset, count)};
    }
};

template <typename T, int N>
class $NAME$Viewer : public $NAME$ViewerBase<T, N>
{
    using Base = $NAME$ViewerBase<T, N>;
    MUDA_VIEWER_COMMON_NAME($NAME$Viewer);

  public:
    using Base::Base;
    MUDA_GENERIC $NAME$Viewer(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC $NAME$Viewer(const C$NAME$Viewer<T, N>&) = delete;

    MUDA_GENERIC $NAME$Viewer<T, N> subview(int offset, int count) const
    {
        return $NAME$Viewer{Base::subview(offset, count)};
    }
};
}  // namespace muda
'''

view_template='''
#include "$NAME$Viewer.h"

namespace muda
{
template <typename T, int N>
class $NAME$ViewBase
{
  public:
    using BlockMatrix = Eigen::Matrix<T, N, N>;

  protected:
    // data
    T* m_data;


  public:
    MUDA_GENERIC $NAME$ViewBase() = default;
    MUDA_GENERIC $NAME$ViewBase(T* data) {}

    MUDA_GENERIC $NAME$ViewBase<T, N> subview(int offset, int count) const
    {
		return $NAME$ViewBase{};
    }

    auto cviewer() const { return C$NAME$Viewer<T, N>{}; }
};

template <typename T, int N>
class C$NAME$View : public $NAME$ViewBase<T, N>
{
    using Base = $NAME$ViewBase<T, N>;

  public:
    using Base::Base;
    MUDA_GENERIC C$NAME$View(const T* data)
        : Base(const_cast<T*>(data))
    {
    }

    MUDA_GENERIC C$NAME$View(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC C$NAME$View<T, N> subview(int offset, int count) const
    {
        return C$NAME$View{Base::subview(offset, count)};
    }

    using Base::cviewer;
};

template <typename T, int N>
class $NAME$View : public $NAME$ViewBase<T, N>
{
    using Base = $NAME$ViewBase<T, N>;

  public:
    using Base::Base;
    MUDA_GENERIC $NAME$View(const Base& base)
        : Base(base)
    {
    }

    MUDA_GENERIC $NAME$View(const C$NAME$View<T, N>&) = delete;

    MUDA_GENERIC $NAME$View<T, N> subview(int offset, int count) const
    {
        return $NAME$View{Base::subview(offset, count)};
    }

    using Base::cviewer;
    auto viewer() const { return $NAME$Viewer<T, N>{}; }
};
}  // namespace muda

namespace muda
{
template <typename T, int N>
struct read_only_viewer<$NAME$View<T, N>>
{
    using type = C$NAME$View<T, N>;
};

template <typename T, int N>
struct read_write_viewer<$NAME$View<T, N>>
{
    using type = $NAME$View<T, N>;
};
}  // namespace muda
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate view and viewer')
    parser.add_argument('name', type=str, help='name of the view')
    args = parser.parse_args()

    # make name snake case
    # CamelCase -> camel_case
    name = args.name
    name = name[0].lower() + name[1:]
    name = "".join(map(lambda x: "_" + x.lower() if x.isupper() else x, name))


    f = open(name + "_view.h", "w")
    f.write(view_template.replace("$NAME$", args.name))
    f.close()

    f = open(name + "_viewer.h", "w")
    f.write(viewer_template.replace("$NAME$", args.name))
    f.close()