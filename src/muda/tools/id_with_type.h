#pragma once
#include <cstdint>
#include <ostream>
#undef max
namespace muda {
class IdWithType {
public:
    using type = uint64_t;
    static constexpr auto invalid_id = std::numeric_limits<uint64_t>::max();
    explicit IdWithType(uint64_t value) noexcept : m_value{value} {}
    explicit IdWithType() noexcept : m_value{invalid_id} {}
    uint64_t value() const noexcept { return m_value; }
    friend std::ostream &operator<<(std::ostream &os, const IdWithType &id) {
        os << id.m_value;
        return os;
    }
    friend bool operator==(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs.m_value == rhs.m_value; }
    friend bool operator!=(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs.m_value != rhs.m_value; }
    friend bool operator<(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs.m_value < rhs.m_value; }
    friend bool operator>(const IdWithType &lhs, const IdWithType &rhs) noexcept { return lhs.m_value > rhs.m_value; }

    bool is_valid() const noexcept { return m_value != invalid_id; }

protected:
    uint64_t m_value{invalid_id};
};
}// namespace luisa::compute::graph