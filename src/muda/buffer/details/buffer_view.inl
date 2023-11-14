#include <muda/buffer/buffer_launch.h>
#include <muda/compute_graph/compute_graph_builder.h>

namespace muda
{
template <typename T>
BufferViewBase<T> BufferViewBase<T>::subview(size_t offset, size_t size) const MUDA_NOEXCEPT
{
    if(ComputeGraphBuilder::is_topo_building())
        return BufferViewBase<T>{};  // dummy

    if(size == ~0)
        size = m_size - offset;
    MUDA_ASSERT(offset + size <= m_size,
                "BufferView out of range, size = %d, yours = %d",
                m_size,
                offset + size);
    return BufferViewBase<T>{m_data, m_offset + offset, size};
}

template <typename T>
void BufferView<T>::fill(const T& v)
{
    BufferLaunch()
        .fill(*this, v)  //
        .wait();
}

template <typename T>
void BufferView<T>::copy_from(const BufferView<T>& other)
{
    BufferLaunch()
        .copy(*this, other)  //
        .wait();
}

template <typename T>
void BufferView<T>::copy_from(T* host)
{
    BufferLaunch()
        .copy(*this, host)  //
        .wait();
}

template <typename T>
void BufferViewBase<T>::copy_to(T* host) const
{
    BufferLaunch()
        .copy(host, *this)  //
        .wait();
}

template <typename T>
Dense1D<T> BufferView<T>::viewer() const MUDA_NOEXCEPT
{
    return Dense1D<T>{this->m_data, static_cast<int>(m_size)};
}

template <typename T>
CDense1D<T> BufferViewBase<T>::cviewer() const MUDA_NOEXCEPT
{
    return CDense1D<T>{m_data, static_cast<int>(m_size)};
}
}  // namespace muda

//namespace muda
//{
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT
//{
//    MUDA_ASSERT(dimx * dimy <= v.size(), "dimx=%d, dimy=%d, v.size()=%d\n", dimx, dimy, v.size());
//    return Dense2D<T>(v.data(), dimx, dimy);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimx, int dimy) MUDA_NOEXCEPT
//{
//    MUDA_ASSERT(dimx * dimy <= v.size(), "dimx=%d, dimy=%d, v.size()=%d\n", dimx, dimy, v.size());
//    return CDense2D<T>(v.data(), dimx, dimy);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT
//{
//    return make_dense2D(v.data(), v.size() / dimy, dimy);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, int dimy) MUDA_NOEXCEPT
//{
//    return make_cdense2D(v.data(), v.size() / dimy, dimy);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT
//{
//    MUDA_ASSERT(dim.x * dim.y <= v.size(),
//                "dim.x=%d, dim.y=%d, v.size()=%d\n",
//                dim.x,
//                dim.y,
//                v.size());
//    return make_dense2D(v.data(), dim.x, dim.y);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense2D(BufferView<T>& v, const int2& dim) MUDA_NOEXCEPT
//{
//    MUDA_ASSERT(dim.x * dim.y <= v.size(),
//                "dim.x=%d, dim.y=%d, v.size()=%d\n",
//                dim.x,
//                dim.y,
//                v.size());
//    return make_cdense2D(v.data(), dim.x, dim.y);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dimx * dimy * dimz <= v.size(),
//                       "dimx=%d, dimy=%d, dimz=%d, v.size()=%d\n",
//                       dimx,
//                       dimy,
//                       dimz,
//                       v.size());
//    return Dense3D<T>(v.data(), dimx, dimy, dimz);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, int dimx, int dimy, int dimz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dimx * dimy * dimz <= v.size(),
//                       "dimx=%d, dimy=%d, dimz=%d, v.size()=%d\n",
//                       dimx,
//                       dimy,
//                       dimz,
//                       v.size());
//    return CDense3D<T>(v.data(), dimx, dimy, dimz);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(
//        dimy * dimz <= v.size(), "dimy=%d, dimz=%d, v.size()=%d\n", dimy, dimz, v.size());
//    return make_dense3D(v.data(), v.size() / (dimy * dimz), dimy, dimz);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, int dimy, int dimz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(
//        dimy * dimz <= v.size(), "dimy=%d, dimz=%d, v.size()=%d\n", dimy, dimz, v.size());
//    return make_cdense3D(v.data(), v.size() / (dimy * dimz), dimy, dimz);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dimyz.x * dimyz.y <= v.size(),
//                       "dimy=%d, dimz=%d, v.size()=%d\n",
//                       dimyz.x,
//                       dimyz.y,
//                       v.size());
//    return make_dense3D(v.data(), v.size() / (dimyz.x * dimyz.y), dimyz.x, dimyz.y);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int2& dimyz) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dimyz.x * dimyz.y <= v.size(),
//                       "dimy=%d, dimz=%d, v.size()=%d\n",
//                       dimyz.x,
//                       dimyz.y,
//                       v.size());
//    return make_cdense3D(
//        v.data(), v.size() / (dimyz.x * dimyz.y), dimyz.x, dimyz.y);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_dense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dim.x * dim.y * dim.z <= v.size(),
//                       "dim.x=%d, dim.y=%d, dim.z=%d, v.size()=%d\n",
//                       dim.x,
//                       dim.y,
//                       dim.z,
//                       v.size());
//    return make_dense3D(v.data(), dim.x, dim.y, dim.z);
//}
//
//template <typename T>
//MUDA_INLINE MUDA_HOST auto make_cdense3D(BufferView<T>& v, const int3& dim) MUDA_NOEXCEPT
//{
//    MUDA_KERNEL_ASSERT(dim.x * dim.y * dim.z <= v.size(),
//                       "dim.x=%d, dim.y=%d, dim.z=%d, v.size()=%d\n",
//                       dim.x,
//                       dim.y,
//                       dim.z,
//                       v.size());
//    return make_cdense3D(v.data(), dim.x, dim.y, dim.z);
//}
//}  // namespace muda