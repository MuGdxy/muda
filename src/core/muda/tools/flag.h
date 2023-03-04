#pragma once
#include <type_traits>
#include <muda/muda_def.h>

namespace muda
{
template <typename FlagBitsType>
struct is_flag_type
{
    static MUDA_CONSTEXPR bool value = false;
};

template <typename FlagBitsType>
MUDA_CONSTEXPR bool is_flag_type_v = is_flag_type<FlagBitsType>::value;

template <typename BitType>
class flags
{
  public:
    using bit_type = BitType;
	using mask_type = typename std::underlying_type<bit_type>::type;
	
    // constructors
    MUDA_CONSTEXPR flags() MUDA_NOEXCEPT : m_mask(0) {}

    MUDA_CONSTEXPR flags(BitType bit) MUDA_NOEXCEPT
        : m_mask(static_cast<mask_type>(bit))
    {
    }

    MUDA_CONSTEXPR flags(flags<BitType> const& rhs) MUDA_NOEXCEPT = default;

    MUDA_CONSTEXPR explicit flags(mask_type flags) MUDA_NOEXCEPT : m_mask(flags)
    {
    }

    // relational operators
    // auto operator<=>(Flags<BitType> const&) const = default;

    MUDA_CONSTEXPR bool operator<(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask < rhs.m_mask;
    }

    MUDA_CONSTEXPR bool operator<=(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask <= rhs.m_mask;
    }

    MUDA_CONSTEXPR bool operator>(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask > rhs.m_mask;
    }

    MUDA_CONSTEXPR bool operator>=(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask >= rhs.m_mask;
    }

    MUDA_CONSTEXPR bool operator==(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask == rhs.m_mask;
    }

    MUDA_CONSTEXPR bool operator!=(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return m_mask != rhs.m_mask;
    }


    // logical operator
    MUDA_CONSTEXPR bool operator!() const MUDA_NOEXCEPT
    {
        return !m_mask;
    }

    // bitwise operators
    MUDA_CONSTEXPR flags<BitType> operator&(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return flags<BitType>(m_mask & rhs.m_mask);
    }

    MUDA_CONSTEXPR flags<BitType> operator|(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return flags<BitType>(m_mask | rhs.m_mask);
    }

    MUDA_CONSTEXPR flags<BitType> operator^(flags<BitType> const& rhs) const MUDA_NOEXCEPT
    {
        return flags<BitType>(m_mask ^ rhs.m_mask);
    }

    MUDA_CONSTEXPR flags<BitType> operator~() const MUDA_NOEXCEPT
    {
        return flags<BitType>(m_mask ^ is_flag_type<BitType>::allFlags.m_mask);
    }

    // assignment operators
    MUDA_CONSTEXPR flags<BitType>& operator=(flags<BitType> const& rhs) MUDA_NOEXCEPT = default;

    MUDA_CONSTEXPR flags<BitType>& operator|=(flags<BitType> const& rhs) MUDA_NOEXCEPT
    {
        m_mask |= rhs.m_mask;
        return *this;
    }

    MUDA_CONSTEXPR flags<BitType>& operator&=(flags<BitType> const& rhs) MUDA_NOEXCEPT
    {
        m_mask &= rhs.m_mask;
        return *this;
    }

    MUDA_CONSTEXPR flags<BitType>& operator^=(flags<BitType> const& rhs) MUDA_NOEXCEPT
    {
        m_mask ^= rhs.m_mask;
        return *this;
    }

    // cast operators
    explicit MUDA_CONSTEXPR operator bool() const MUDA_NOEXCEPT
    {
        return !!m_mask;
    }

    explicit MUDA_CONSTEXPR operator mask_type() const MUDA_NOEXCEPT
    {
        return m_mask;
    }

  private:
    mask_type m_mask;
};

//// relational operators only needed for pre C++20
//template <typename BitType>
//MUDA_CONSTEXPR bool operator<(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator>(bit);
//}
//
//template <typename BitType>
//MUDA_CONSTEXPR bool operator<=(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator>=(bit);
//}
//
//template <typename BitType>
//MUDA_CONSTEXPR bool operator>(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator<(bit);
//}
//
//template <typename BitType>
//MUDA_CONSTEXPR bool operator>=(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator<=(bit);
//}
//
//template <typename BitType>
//MUDA_CONSTEXPR bool operator==(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator==(bit);
//}
//
//template <typename BitType>
//MUDA_CONSTEXPR bool operator!=(BitType bit, Flags<BitType> const& flags) MUDA_NOEXCEPT
//{
//    return flags.operator!=(bit);
//}


// bitwise operators
template <typename BitType>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator&(BitType bit, flags<BitType> const& flags) MUDA_NOEXCEPT
{
    return flags.operator&(bit);
}

template <typename BitType>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator|(BitType bit, flags<BitType> const& flags) MUDA_NOEXCEPT
{
    return flags.operator|(bit);
}

template <typename BitType>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator^(BitType bit, flags<BitType> const& flags) MUDA_NOEXCEPT
{
    return flags.operator^(bit);
}

// bitwise operators on BitType
template <typename BitType, typename std::enable_if<is_flag_type<BitType>::isBitmask, bool>::type = true>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator&(BitType lhs, BitType rhs) MUDA_NOEXCEPT
{
    return flags<BitType>(lhs) & rhs;
}

template <typename BitType, typename std::enable_if<is_flag_type<BitType>::isBitmask, bool>::type = true>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator|(BitType lhs, BitType rhs) MUDA_NOEXCEPT
{
    return flags<BitType>(lhs) | rhs;
}

template <typename BitType, typename std::enable_if<is_flag_type<BitType>::isBitmask, bool>::type = true>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator^(BitType lhs, BitType rhs) MUDA_NOEXCEPT
{
    return flags<BitType>(lhs) ^ rhs;
}

template <typename BitType, typename std::enable_if<is_flag_type<BitType>::isBitmask, bool>::type = true>
MUDA_INLINE MUDA_CONSTEXPR flags<BitType> operator~(BitType bit) MUDA_NOEXCEPT
{
    return ~(flags<BitType>(bit));
}
}


