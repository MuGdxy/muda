#pragma once

namespace muda
{
enum class ComputeGraphPhase
{
    None,
    DummyBuilding,
    Building,
    Updating,
    SerialLaunching,
    Max
};
}