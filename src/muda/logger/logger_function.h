#pragma once
#include <muda/logger/logger.h>

namespace muda
{
// signed
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, char1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, char2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, char3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, char4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, short1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, short2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, short3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, short4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, int1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, int2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, int3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, int4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, long1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, long2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, long3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, long4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, longlong1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, longlong2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, longlong3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, longlong4 val);

// unsigned
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uchar1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uchar2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uchar3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uchar4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uint1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uint2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uint3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, uint4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulong1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulong2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulong3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulong4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulonglong1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulonglong2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulonglong3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, ulonglong4 val);

// float
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, float1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, float2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, float3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, float4 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, double1 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, double2 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, double3 val);
MUDA_INLINE MUDA_DEVICE LogProxy& operator<<(LogProxy& proxy, double4 val);
}  // namespace muda
#include <muda/logger/details/logger_function.inl>