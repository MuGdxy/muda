#pragma once
#include <muda/launch/parallel_for.h>
#include <muda/buffer/buffer_view.h>
#include <muda/buffer/buffer_2d_view.h>
#include <muda/buffer/buffer_3d_view.h>

namespace muda::details::buffer
{
//template <typename T>
//MUDA_DEVICE void device_destruct(T* t)
//{
//    t->~T();
//}
//
//template <typename T>
//MUDA_DEVICE void device_construct(T* t)
//{
//    new(t) T();
//}
//
//template <typename T, typename F>
//MUDA_HOST void kernel_for_each(
//    int grid_dim, int block_dim, cudaStream_t stream, BufferView<T> buffer_view, F func)
//{
//    ParallelFor(grid_dim, block_dim, 0, stream)
//        .apply(buffer_view.size(),
//               [buffer_view, func] __device__(int i) mutable
//               { func(buffer_view.data(i)); })
//        .wait();
//}
//
//// for_each 2D
//template <typename T, typename F>
//MUDA_HOST void kernel_for_each(int             grid_dim,
//                               int             block_dim,
//                               cudaStream_t    stream,
//                               Buffer2DView<T> buffer_view,
//                               F               func)
//{
//    ParallelFor(grid_dim, block_dim, 0, stream)
//        .apply(buffer_view.total_size(),
//               [buffer_view, func] __device__(int i) mutable
//               { func(buffer_view.data(i)); });
//}
//
//// for_each 3D
//template <typename T, typename F>
//MUDA_HOST void kernel_for_each(int             grid_dim,
//                               int             block_dim,
//                               cudaStream_t    stream,
//                               Buffer3DView<T> buffer_view,
//                               F               func)
//{
//    ParallelFor(grid_dim, block_dim, 0, stream)
//        .apply(buffer_view.total_size(),
//               [buffer_view, func] __device__(int i) mutable
//               { func(buffer_view.data(i)); });
//}

// destruct 0D
template <typename T>
MUDA_HOST void kernel_destruct(cudaStream_t stream, VarView<T> view)
{
    ParallelFor(1, 1, 0, stream)
        .apply(1, [view] __device__(int i) mutable { view.data()->~T(); });
}

// destruct 1D
template <typename T>
MUDA_HOST void kernel_destruct(int grid_dim, int block_dim, cudaStream_t stream, BufferView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_destruct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.size(),
               [buffer_view] __device__(int i) mutable
               { buffer_view.data(i)->~T(); });
}

// destruct 2D
template <typename T>
MUDA_HOST void kernel_destruct(int grid_dim, int block_dim, cudaStream_t stream, Buffer2DView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_destruct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.total_size(),
               [buffer_view] __device__(int i) mutable
               { buffer_view.data(i)->~T(); });
}

// destruct 3D
template <typename T>
MUDA_HOST void kernel_destruct(int grid_dim, int block_dim, cudaStream_t stream, Buffer3DView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_destruct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.total_size(),
               [buffer_view] __device__(int i) mutable
               { buffer_view.data(i)->~T(); });
}

// construct 0D
template <typename T>
MUDA_HOST void kernel_construct(cudaStream_t stream, VarView<T> view)
{
    ParallelFor(1, 1, 0, stream)
        .apply(1, [view] __device__(int i) mutable { new(view.data()) T(); });
}

// construct 1D
template <typename T>
MUDA_HOST void kernel_construct(int grid_dim, int block_dim, cudaStream_t stream, BufferView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_construct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.size(),
               [buffer_view] __device__(int i) mutable
               { new(buffer_view.data(i)) T(); });
}

// construct 2D
template <typename T>
MUDA_HOST void kernel_construct(int grid_dim, int block_dim, cudaStream_t stream, Buffer2DView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_construct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.total_size(),
               [buffer_view] __device__(int i) mutable
               { new(buffer_view.data(i)) T(); });
}

// construct 3D
template <typename T>
MUDA_HOST void kernel_construct(int grid_dim, int block_dim, cudaStream_t stream, Buffer3DView<T> buffer_view)
{
    // kernel_for_each(grid_dim, block_dim, stream, buffer_view, device_construct<T>);
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(buffer_view.total_size(),
               [buffer_view] __device__(int i) mutable
               { new(buffer_view.data(i)) T(); });
}

// assign 0D
template <typename T>
MUDA_HOST void kernel_assign(cudaStream_t stream, VarView<T> dst, CVarView<T> src)
{
    ParallelFor(1, 1, 0, stream)
        .apply(1,
               [dst, src] __device__(int i) mutable
               { *dst.data() = *src.data(); });
}

// assign 1D
template <typename T>
MUDA_HOST void kernel_assign(int            grid_dim,
                             int            block_dim,
                             cudaStream_t   stream,
                             BufferView<T>  dst,
                             CBufferView<T> src)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.size(),
               [dst, src] __device__(int i) mutable
               { *dst.data(i) = *src.data(i); });
}

// assign 2D
template <typename T>
MUDA_HOST void kernel_assign(int              grid_dim,
                             int              block_dim,
                             cudaStream_t     stream,
                             Buffer2DView<T>  dst,
                             CBuffer2DView<T> src)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, src] __device__(int i) mutable
               { *dst.data(i) = *src.data(i); });
}

// assign 3D
template <typename T>
MUDA_HOST void kernel_assign(int              grid_dim,
                             int              block_dim,
                             cudaStream_t     stream,
                             Buffer3DView<T>  dst,
                             CBuffer3DView<T> src)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, src] __device__(int i) mutable
               { *dst.data(i) = *src.data(i); });
}

// fill 0D
template <typename T>
MUDA_HOST void kernel_fill(cudaStream_t stream, VarView<T> dst, const T& val)
{
    ParallelFor(1, 1, 0, stream)
        .apply(1, [dst, val] __device__(int i) mutable { *dst.data() = val; });
}

// fill 1D
template <typename T>
MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, BufferView<T> dst, const T& val)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
}

// fill 2D
template <typename T>
MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, Buffer2DView<T> dst, const T& val)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
};

// fill 3D
template <typename T>
MUDA_HOST void kernel_fill(
    int grid_dim, int block_dim, cudaStream_t stream, Buffer3DView<T> dst, const T& val)
{
    ParallelFor(grid_dim, block_dim, 0, stream)
        .apply(dst.total_size(),
               [dst, val] __device__(int i) mutable { *dst.data(i) = val; });
};
}  // namespace muda::details::buffer
