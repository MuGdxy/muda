#include <muda/buffer/device_var.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <muda/buffer/device_buffer_3d.h>

#include <muda/buffer/var_view.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/buffer/buffer_3d_view.h>

#include <muda/buffer/graph_var_view.h>
#include <muda/buffer/graph_buffer_view.h>
#include <muda/buffer/graph_buffer_2d_view.h>
#include <muda/buffer/graph_buffer_3d_view.h>

#include <muda/buffer/agent.h>
#include <muda/buffer/reshape_nd/nd_reshaper.h>

namespace muda
{
/**********************************************************************************************
* 
* Buffer API
* 0D DeviceVar
* 1D DeviceBuffer
* 2D DeviceBuffer2D
* 3D DeviceBuffer3D
* 
**********************************************************************************************/
template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size)
{
    return resize(
        buffer,
        new_size,
        [&](BufferView<T> view)  // construct
        {
            if constexpr(std::is_trivially_constructible_v<T>)
            {
                Memory(m_stream).set(view.data(), view.size() * sizeof(T), 0);
            }
            else
            {
                static_assert(std::is_constructible_v<T>,
                              "The type T must be constructible, which means T must have a 0-arg constructor");

                details::buffer::kernel_construct(m_grid_dim, m_block_dim, m_stream, view);
            }
        });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer2D<T>& buffer, Extent2D extent)
{
    return resize(buffer,
                  extent,
                  [&](Buffer2DView<T> view)  // construct
                  {
                      // cudaMemset2D has no offset, so we can't use it here

                      //if constexpr(std::is_trivially_constructible_v<T>)
                      //{
                      //    Extent2D extent = view.extent();
                      //    Memory(m_stream).set(view.data(),
                      //                         view.pitch_bytes(),
                      //                         extent.width() * sizeof(T),
                      //                         extent.height(),
                      //                         0);
                      //}
                      //else
                      //{
                      //    static_assert(std::is_constructible_v<T>,
                      //                  "The type T must be constructible, which means T must have a 0-arg constructor");

                      //    details::buffer::kernel_construct(m_grid_dim, m_block_dim, m_stream, view);
                      //}
                      details::buffer::kernel_construct(m_grid_dim, m_block_dim, m_stream, view);
                  });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer3D<T>& buffer, Extent3D extent)
{
    return resize(buffer,
                  extent,
                  [&](Buffer3DView<T> view)  // construct
                  {
                      // cudaMemset3D has no offset, so we can't use it here

                      //if constexpr(std::is_trivially_constructible_v<T>)
                      //{
                      //    Extent3D       extent      = view.extent();
                      //    cudaPitchedPtr pitched_ptr = view.cuda_pitched_ptr();
                      //    Memory(m_stream).set(pitched_ptr, extent.cuda_extent<T>(), 0);
                      //}
                      //else
                      //{
                      //    static_assert(std::is_constructible_v<T>,
                      //                  "The type T must be constructible, which means T must have a 0-arg constructor");
                      //    details::buffer::kernel_construct(m_grid_dim, m_block_dim, m_stream, view);
                      //
                      //}
                      details::buffer::kernel_construct(m_grid_dim, m_block_dim, m_stream, view);
                  });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size, const T& val)
{
    return resize(buffer, new_size, [&](BufferView<T> view) { fill(view, val); });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer2D<T>& buffer,
                                             Extent2D           extent,
                                             const T&           val)
{
    return resize(buffer, extent, [&](Buffer2DView<T> view) { fill(view, val); });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer3D<T>& buffer,
                                             Extent3D           extent,
                                             const T&           val)
{
    return resize(buffer, extent, [&](Buffer3DView<T> view) { fill(view, val); });
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::clear(DeviceBuffer<T>& buffer)
{
    resize(buffer, 0);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::clear(DeviceBuffer2D<T>& buffer)
{
    resize(buffer, Extent2D::Zero());
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::clear(DeviceBuffer3D<T>& buffer)
{
    resize(buffer, Extent3D::Zero());
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::alloc(DeviceBuffer<T>& buffer, size_t n)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot alloc a buffer in a compute graph");
    MUDA_ASSERT(!buffer.m_data, "The buffer is already allocated");
    resize(buffer, n);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::alloc(DeviceBuffer2D<T>& buffer, Extent2D extent)
{
    MUDA_ASSERT(!buffer.m_data, "The buffer is already allocated");
    resize(buffer, extent);
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::alloc(DeviceBuffer3D<T>& buffer, Extent3D extent)
{
    MUDA_ASSERT(!buffer.m_data, "The buffer is already allocated");
    resize(buffer, extent);
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::free(DeviceBuffer<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot free a buffer in a compute graph");
    MUDA_ASSERT(buffer.m_data, "The buffer is not allocated");

    auto& m_data     = buffer.m_data;
    auto& m_size     = buffer.m_size;
    auto& m_capacity = buffer.m_capacity;

    Memory(m_stream).free(m_data);
    m_data     = nullptr;
    m_size     = 0;
    m_capacity = 0;
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::free(DeviceBuffer2D<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot free a buffer in a compute graph");
    MUDA_ASSERT(buffer.m_data, "The buffer is not allocated");

    auto& m_data        = buffer.m_data;
    auto& m_pitch_bytes = buffer.m_pitch_bytes;
    auto& m_extent      = buffer.m_extent;
    auto& m_capacity    = buffer.m_capacity;

    Memory(m_stream).free(m_data);
    m_data        = nullptr;
    m_pitch_bytes = 0;
    m_extent      = Extent2D::Zero();
    m_capacity    = Extent2D::Zero();
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::free(DeviceBuffer3D<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot free a buffer in a compute graph");
    MUDA_ASSERT(buffer.m_data, "The buffer is not allocated");

    auto& m_data             = buffer.m_data;
    auto& m_pitch_bytes      = buffer.m_pitch_bytes;
    auto& m_pitch_bytes_area = buffer.m_pitch_bytes_area;
    auto& m_extent           = buffer.m_extent;
    auto& m_capacity         = buffer.m_capacity;

    Memory(m_stream).free(m_data);
    m_data             = nullptr;
    m_pitch_bytes      = 0;
    m_pitch_bytes_area = 0;
    m_extent           = Extent3D::Zero();
    m_capacity         = Extent3D::Zero();
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::shrink_to_fit(DeviceBuffer<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot shrink a buffer in a compute graph");

    auto  mem        = Memory(m_stream);
    auto& m_data     = buffer.m_data;
    auto& m_size     = buffer.m_size;
    auto& m_capacity = buffer.m_capacity;
    if(m_size < m_capacity)
    {
        T* ptr = nullptr;
        if(m_size > 0)
        {
            mem.alloc(&ptr, m_size * sizeof(T));
            BufferView<T> dst{ptr, 0, m_size};
            copy<T>(dst, buffer.view());
        }
        if(m_data)
            mem.free(m_data);
        m_data     = ptr;
        m_capacity = m_size;
    }
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::shrink_to_fit(DeviceBuffer2D<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot shrink a buffer in a compute graph");

    auto  mem           = Memory(m_stream);
    auto& m_data        = buffer.m_data;
    auto& m_pitch_bytes = buffer.m_pitch_bytes;
    auto& m_extent      = buffer.m_extent;
    auto& m_capacity    = buffer.m_capacity;

    if(m_extent < m_capacity)
    {
        T* ptr = nullptr;
        if(!(m_extent == Extent2D::Zero()))
        {
            size_t new_pitch_bytes = ~0;
            mem.alloc_2d(
                &ptr, &new_pitch_bytes, m_extent.width() * sizeof(T), m_extent.height());

            Buffer2DView<T> dst{ptr, new_pitch_bytes, Offset2D::Zero(), m_extent};
            copy<T>(dst, buffer.view());

            m_pitch_bytes = new_pitch_bytes;
        }
        if(m_data)
            mem.free(m_data);

        m_data     = ptr;
        m_capacity = m_extent;
    }
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::shrink_to_fit(DeviceBuffer3D<T>& buffer)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot shrink a buffer in a compute graph");

    auto  mem                = Memory(m_stream);
    auto& m_data             = buffer.m_data;
    auto& m_pitch_bytes      = buffer.m_pitch_bytes;
    auto& m_pitch_bytes_area = buffer.m_pitch_bytes_area;
    auto& m_extent           = buffer.m_extent;
    auto& m_capacity         = buffer.m_capacity;

    if(m_extent < m_capacity)
    {
        T* ptr = nullptr;
        if(!(m_extent == Extent3D::Zero()))
        {

            cudaPitchedPtr pitched_ptr;
            mem.alloc_3d(&pitched_ptr, m_extent.template cuda_extent<T>());
            ptr                         = reinterpret_cast<T*>(pitched_ptr.ptr);
            size_t new_pitch_bytes      = pitched_ptr.pitch;
            size_t new_pitch_bytes_area = new_pitch_bytes * m_extent.height();

            Buffer3DView<T> dst{
                ptr, new_pitch_bytes, new_pitch_bytes_area, Offset3D::Zero(), m_extent};

            copy<T>(dst, buffer.view());

            m_pitch_bytes      = new_pitch_bytes;
            m_pitch_bytes_area = new_pitch_bytes_area;
        }
        if(m_data)
            mem.free(m_data);

        m_data     = ptr;
        m_capacity = m_extent;
    }
    return *this;
}


/**********************************************************************************************
* 
* BufferView Copy: Device <- Device
* 
**********************************************************************************************/
template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(VarView<T> dst, CVarView<T> src)
{
    details::buffer::kernel_assign(m_grid_dim, m_block_dim, m_stream, dst, src);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(BufferView<T> dst, CBufferView<T> src)
{
    MUDA_ASSERT(dst.size() == src.size(), "BufferView should have the same size");
    details::buffer::kernel_assign(m_grid_dim, m_block_dim, m_stream, dst, src);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer2DView<T> dst, CBuffer2DView<T> src)
{
    MUDA_ASSERT(dst.extent() == src.extent(), "BufferView should have the same size");
    details::buffer::kernel_assign(m_grid_dim, m_block_dim, m_stream, dst, src);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer3DView<T> dst, CBuffer3DView<T> src)
{
    MUDA_ASSERT(dst.extent() == src.extent(), "BufferView should have the same size");
    details::buffer::kernel_assign(m_grid_dim, m_block_dim, m_stream, dst, src);
    return *this;
}


template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(VarView<T> dst, VarView<T> src)
{
    return copy(dst, src.operator CVarView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(BufferView<T> dst, BufferView<T> src)
{
    return copy(dst, src.operator CBufferView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer2DView<T> dst, Buffer2DView<T> src)
{
    return copy(dst, src.operator CBuffer2DView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer3DView<T> dst, Buffer3DView<T> src)
{
    return copy(dst, src.operator CBuffer3DView<T>());
}


template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<VarView<T>>& dst,
                                           const ComputeGraphVar<VarView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<BufferView<T>>& dst,
                                           const ComputeGraphVar<BufferView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<Buffer2DView<T>>& dst,
                                           const ComputeGraphVar<Buffer2DView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<Buffer3DView<T>>& dst,
                                           const ComputeGraphVar<Buffer3DView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}


/**********************************************************************************************
* 
* BufferView Copy: Host <- Device
* 
**********************************************************************************************/
template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, CVarView<T> src)
{
    Memory(m_stream).download(dst, src.data(), sizeof(T));
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, CBufferView<T> src)
{
    Memory(m_stream).download(dst, src.data(), src.size() * sizeof(T));
    return *this;
}


template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, CBuffer2DView<T> src)
{
    cudaMemcpy3DParms parms = {0};

    parms.srcPtr = src.cuda_pitched_ptr();
    parms.srcPos = src.offset().template cuda_pos<T>();
    parms.dstPtr = make_cudaPitchedPtr(
        dst, parms.srcPtr.xsize, parms.srcPtr.xsize, parms.srcPtr.ysize);
    parms.extent = src.extent().template cuda_extent<T>();
    parms.dstPos = make_cudaPos(0, 0, 0);

    Memory(m_stream).download(parms);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, CBuffer3DView<T> src)
{
    cudaMemcpy3DParms parms = {0};

    parms.srcPtr = src.cuda_pitched_ptr();
    parms.srcPos = src.offset().template cuda_pos<T>();
    parms.dstPtr = make_cudaPitchedPtr(
        dst, parms.srcPtr.xsize, parms.srcPtr.xsize, parms.srcPtr.ysize);
    parms.extent = src.extent().template cuda_extent<T>();
    parms.dstPos = make_cudaPos(0, 0, 0);

    Memory(m_stream).download(parms);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, VarView<T> src)
{
    return copy(dst, src.operator CVarView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, BufferView<T> src)
{
    return copy(dst, src.operator CBufferView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, Buffer2DView<T> src)
{
    return copy(dst, src.operator CBuffer2DView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(T* dst, Buffer3DView<T> src)
{
    return copy(dst, src.operator CBuffer3DView<T>());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<T*>& dst,
                                           const ComputeGraphVar<VarView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<T*>& dst,
                                           const ComputeGraphVar<BufferView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<T*>& dst,
                                           const ComputeGraphVar<Buffer2DView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<T*>& dst,
                                           const ComputeGraphVar<Buffer3DView<T>>& src)
{
    return copy(dst.eval(), src.ceval());
}

/**********************************************************************************************
* 
* BufferView Copy: Device <- Host
* 
**********************************************************************************************/
template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(VarView<T> dst, const T* src)
{
    Memory(m_stream).upload(dst.data(), src, sizeof(T));
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(BufferView<T> dst, const T* src)
{
    Memory(m_stream).upload(dst.data(), src, dst.size() * sizeof(T));
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer2DView<T> dst, const T* src)
{
    cudaMemcpy3DParms parms = {0};

    parms.extent = dst.extent().template cuda_extent<T>();
    parms.dstPos = dst.offset().template cuda_pos<T>();

    parms.srcPtr = make_cudaPitchedPtr(const_cast<T*>(src),
                                       parms.dstPtr.xsize,
                                       parms.dstPtr.xsize,
                                       parms.dstPtr.ysize);
    parms.srcPos = make_cudaPos(0, 0, 0);

    parms.dstPtr = dst.cuda_pitched_ptr();

    Memory(m_stream).upload(parms);

    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(Buffer3DView<T> dst, const T* src)
{
    cudaMemcpy3DParms parms = {0};

    parms.extent = dst.extent().template cuda_extent<T>();
    parms.dstPos = dst.offset().template cuda_pos<T>();

    parms.srcPtr = make_cudaPitchedPtr(const_cast<T*>(src),
                                       parms.dstPtr.xsize,
                                       parms.dstPtr.xsize,
                                       parms.dstPtr.ysize);
    parms.srcPos = make_cudaPos(0, 0, 0);

    parms.dstPtr = dst.cuda_pitched_ptr();

    Memory(m_stream).upload(parms);

    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<VarView<T>>& dst,
                                           const ComputeGraphVar<T*>&   src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<BufferView<T>>& dst,
                                           const ComputeGraphVar<T*>&      src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::copy(ComputeGraphVar<Buffer2DView<T>>& dst,
                                           const ComputeGraphVar<T*>& src)
{
    return copy(dst.eval(), src.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& copy(ComputeGraphVar<Buffer3DView<T>>& dst,
                             const ComputeGraphVar<T*>&        src)
{
    return copy(dst.eval(), src.ceval());
}

/**********************************************************************************************
* 
* BufferView Scatter: Device <- Host
* 
**********************************************************************************************/
template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(VarView<T> view, const T& val)
{
    details::buffer::kernel_fill(m_stream, view, val);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(BufferView<T> buffer, const T& val)
{
    details::buffer::kernel_fill(m_grid_dim, m_block_dim, m_stream, buffer, val);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(Buffer2DView<T> buffer, const T& val)
{
    details::buffer::kernel_fill(m_grid_dim, m_block_dim, m_stream, buffer, val);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(Buffer3DView<T> buffer, const T& val)
{
    details::buffer::kernel_fill(m_grid_dim, m_block_dim, m_stream, buffer, val);
    return *this;
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(ComputeGraphVar<VarView<T>>& buffer,
                                           const ComputeGraphVar<T>&    val)
{
    return fill(buffer.eval(), val.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(ComputeGraphVar<BufferView<T>>& buffer,
                                           const ComputeGraphVar<T>& val)
{
    return fill(buffer.eval(), val.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(ComputeGraphVar<Buffer2DView<T>>& buffer,
                                           const ComputeGraphVar<T>& val)
{
    return fill(buffer.eval(), val.ceval());
}

template <typename T>
MUDA_HOST BufferLaunch& BufferLaunch::fill(ComputeGraphVar<Buffer3DView<T>>& buffer,
                                           const ComputeGraphVar<T>& val)
{
    return fill(buffer.eval(), val.ceval());
}

/**********************************************************************************************
* 
* Internal BufferView Resize
* 
**********************************************************************************************/
template <typename T, typename FConstruct>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer<T>& buffer, size_t new_size, FConstruct&& fct)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot resize a buffer in a compute graph");
    details::buffer::NDReshaper::resize(
        m_grid_dim, m_block_dim, m_stream, buffer, new_size, std::forward<FConstruct>(fct));
    return *this;
}

template <typename T, typename FConstruct>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer2D<T>& buffer,
                                             Extent2D           new_extent,
                                             FConstruct&&       fct)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot resize a buffer in a compute graph");

    auto mem = Memory(m_stream);

    auto& m_data        = buffer.m_data;
    auto& m_pitch_bytes = buffer.m_pitch_bytes;
    auto& m_extent      = buffer.m_extent;
    auto& m_capacity    = buffer.m_capacity;

    if(new_extent == m_extent)
        return *this;

    auto old_extent = m_extent;

    if(new_extent < m_extent)
    {
        // if the new extent is smaller than the old extent in all dimensions
        // destruct the old memory
        if constexpr(!std::is_trivially_destructible_v<T>)
        {
            Offset2D offset{old_extent.height(), old_extent.width()};
            auto     to_destruct = buffer.view(offset);
            details::buffer::kernel_destruct(m_grid_dim, m_block_dim, m_stream, to_destruct);
        }
        m_extent = new_extent;
        return *this;
    }

    if(new_extent <= m_capacity)
    {
        // all dimensions are bigger then the new extent
        Offset2D offset{old_extent.height(), old_extent.width()};
        auto     to_construct = buffer.view(offset, new_extent);
        fct(to_construct);
        m_extent = new_extent;
    }
    else
    {
        // at least one dimension is smaller than the new extent
        // so we need to allocate a new buffer (m_capacity)
        // which is bigger than the new_extent in all dimensions
        auto   new_capacity = max(new_extent, m_capacity);
        T*     ptr;
        size_t new_pitch_bytes;
        mem.alloc_2d(&ptr,
                     &new_pitch_bytes,
                     sizeof(T) * new_capacity.width(),
                     new_capacity.height());

        // if the old buffer was allocated, copy old data
        if(m_data)
        {
            Buffer2DView<T> dst{ptr, new_pitch_bytes, Offset2D::Zero(), old_extent};
            copy<T>(dst, buffer.view());
        }

        // construct the new memory
        {
            if(old_extent == Extent2D::Zero())
            {
                Buffer2DView<T> to_construct{ptr, new_pitch_bytes, Offset2D::Zero(), new_extent};
                fct(to_construct);
            }
            else if(old_extent.width() == new_extent.width())
            {
                // there are 2 blocks:
                //tex:
                //$$
                //\begin {bmatrix}
                // O  \\
                // N
                //\end {bmatrix}
                //$$
                Offset2D offset{old_extent.height(), 0};
                Buffer2DView<T> to_construct{ptr, new_pitch_bytes, offset, new_extent};
                fct(to_construct);
            }
            else if(old_extent.height() == new_extent.height())
            {
            }
        }

        // if the old buffer was allocated, deallocate it
        if(m_data)
            mem.free(m_data);

        m_data        = ptr;
        m_pitch_bytes = new_pitch_bytes;
        m_extent      = new_extent;
        m_capacity    = new_capacity;
    }
    return *this;
}
//using T          = float;
//using FConstruct = std::function<void(Buffer3DView<T>)>;
template <typename T, typename FConstruct>
MUDA_HOST BufferLaunch& BufferLaunch::resize(DeviceBuffer3D<T>& buffer,
                                             Extent3D           new_extent,
                                             FConstruct&&       fct)
{
    MUDA_ASSERT(ComputeGraphBuilder::is_direct_launching(),
                "cannot resize a buffer in a compute graph");

    auto mem = Memory(m_stream);

    auto& m_data             = buffer.m_data;
    auto& m_pitch_bytes      = buffer.m_pitch_bytes;
    auto& m_pitch_bytes_area = buffer.m_pitch_bytes_area;
    auto& m_extent           = buffer.m_extent;
    auto& m_capacity         = buffer.m_capacity;

    if(new_extent == m_extent)
        return *this;

    auto old_extent = m_extent;

    if(new_extent < m_extent)
    {
        // if the new extent is smaller than the old extent in all dimensions
        // destruct the old memory
        if constexpr(!std::is_trivially_destructible_v<T>)
        {
            Offset3D offset{old_extent.depth(), old_extent.height(), old_extent.width()};
            auto to_destruct = buffer.view(offset);
            details::buffer::kernel_destruct(m_grid_dim, m_block_dim, m_stream, to_destruct);
        }
        m_extent = new_extent;
        return *this;
    }

    if(new_extent <= m_capacity)
    {
        // all dimensions are bigger then the new extent
        Offset3D offset{old_extent.depth(), old_extent.height(), old_extent.width()};
        auto to_construct = buffer.view(offset, new_extent);
        fct(to_construct);
        m_extent = new_extent;
    }
    else
    {
        // at least one dimension is smaller than the new extent
        // so we need to allocate a new buffer (m_capacity)
        // which is bigger than the new_extent in all dimensions
        auto           new_capacity = max(new_extent, m_capacity);
        cudaPitchedPtr pitched_ptr;

        mem.alloc_3d(&pitched_ptr, new_capacity.template cuda_extent<T>());
        T*     ptr                  = reinterpret_cast<T*>(pitched_ptr.ptr);
        size_t new_pitch_bytes      = pitched_ptr.pitch;
        size_t new_pitch_bytes_area = new_pitch_bytes * new_capacity.height();

        // if the old buffer was allocated, copy old data
        if(m_data)
        {
            Buffer3DView<T> dst{
                ptr, new_pitch_bytes, new_pitch_bytes_area, Offset3D::Zero(), old_extent};
            copy<T>(dst, buffer.view());
        }

        // construct the new memory
        {
            Offset3D offset{old_extent.depth(), old_extent.height(), old_extent.width()};
            Buffer3DView<T> to_construct{
                ptr, new_pitch_bytes, new_pitch_bytes_area, offset, new_extent};
            fct(to_construct);
        }

        // if the old buffer was allocated, deallocate it
        if(m_data)
            mem.free(m_data);

        m_data             = ptr;
        m_pitch_bytes      = new_pitch_bytes;
        m_pitch_bytes_area = new_pitch_bytes_area;
        m_extent           = new_extent;
        m_capacity         = new_capacity;
    }
    return *this;
}
}  // namespace muda