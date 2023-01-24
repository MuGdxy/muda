#pragma once
#include <type_traits>

namespace muda
{
template <class EnumType, class UnderlyingType = typename std::underlying_type<EnumType>::type>
class flag
{
  public:
    flag()
        : m_flag(0)
    {
    }
    flag(EnumType singleFlag)
        : m_flag(singleFlag)
    {
    }
    flag(const flag& original)
        : m_flag(original.m_flag)
    {
    }

    flag& operator|=(EnumType addValue)
    {
        m_flag |= addValue;
        return *this;
    }
    flag operator|(EnumType addValue)
    {
        flag result(*this);
        result |= addValue;
        return result;
    }
    flag& operator&=(EnumType maskValue)
    {
        m_flag &= maskValue;
        return *this;
    }
    flag operator&(EnumType maskValue)
    {
        flag result(*this);
        result &= maskValue;
        return result;
    }
    flag operator~()
    {
        flag result(*this);
        result.m_flag = ~result.m_flag;
        return result;
    }
    explicit operator bool() { return m_flag != 0; }

  protected:
    UnderlyingType m_flag;
};
}