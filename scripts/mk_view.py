view_template='''#include <muda/view/view_base.h>

namespace muda
{
template <bool IsConst, typename Ty>
class $VIEW_NAME$ViewBase : public ViewBase<IsConst>
{
  public:
    static_assert(!std::is_const_v<Ty>, "Ty must be non-const");
    using ConstView                  = $VIEW_NAME$ViewBase<true, Ty>;
    using NonConstView               = $VIEW_NAME$ViewBase<false, Ty>;
    using ThisView                   = $VIEW_NAME$ViewBase<IsConst, Ty>;

  protected:
    // data
    auto_const_t<Ty>* m_values = nullptr;
  public:
    $VIEW_NAME$ViewBase() = default;
    $VIEW_NAME$ViewBase(auto_const_t<IsConst, Ty>*  values)
        : m_values(values)
    {
    }
    
    // explicit conversion to non-const
    ConstView as_const() const
    {
        return ConstView{m_values};
    }

    // implicit conversion to const
    operator ConstView() const
    {
        return as_const();
    }

    // non-const access
    auto_const_t<Ty>* values() { return m_values; }

    // const access
    auto values() const { return m_values; }
};

template <typename Ty>
using $VIEW_NAME$View = $VIEW_NAME$ViewBase<false, Ty>;
template <typename Ty>
using C$VIEW_NAME$View = $VIEW_NAME$ViewBase<true, Ty>;
}  // namespace muda

namespace muda
{
template <typename Ty>
struct read_only_viewer<$VIEW_NAME$View<Ty>>
{
    using type = C$VIEW_NAME$View<Ty>;
};

template <typename Ty>
struct read_write_viewer<$VIEW_NAME$View<Ty>>
{
    using type = $VIEW_NAME$View<Ty>;
};
}  // namespace muda
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate view and viewer')
    parser.add_argument('name', type=str, help='name of the view')
    args = parser.parse_args()

    print(view_template.replace('$VIEW_NAME$', args.name))