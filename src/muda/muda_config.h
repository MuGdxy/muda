#pragma once
namespace muda
{
constexpr bool mudaNoCheck = false;
namespace config
{
    constexpr bool on(bool cond = false)
    {
        return cond && !mudaNoCheck;
    }
}  // namespace config
constexpr bool debugViewers          = config::on(true);
constexpr bool debugTiccd            = config::on(true);
constexpr bool debugThreadOnly = config::on(true);
constexpr bool trapOnError     = config::on(true);
}  // namespace muda

#undef MUDA_NO_CHECKS