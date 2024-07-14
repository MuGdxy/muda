#pragma once

namespace muda
{
template <typename T>
class VarView;

template <typename T>
class CVarView;

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