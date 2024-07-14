#pragma once

namespace muda
{
template <bool IsConst, typename T>
class VarViewT;

template <typename T>
using VarView = VarViewT<false, T>;

template <typename T>
using CVarView = VarViewT<true, T>;

template <bool IsConst, typename T>
class BufferViewT;

template <typename T>
using BufferView = BufferViewT<false, T>;

template <typename T>
using CBufferView = BufferViewT<true, T>;

template <typename T>
class Buffer2DView;

template <typename T>
class CBuffer2DView;

template <typename T>
class Buffer3DView;

template <typename T>
class CBuffer3DView;

template <typename T>
class DeviceVar;

template <typename T>
class DeviceBuffer;

template <typename T>
class DeviceBuffer2D;

template <typename T>
class DeviceBuffer3D;
}  // namespace muda