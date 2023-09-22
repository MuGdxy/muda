#pragma once

namespace muda
{
enum class ComputeGraphVarUsage
{
	None,
	Read,
	ReadWrite,
	Max
};
}  // namespace muda