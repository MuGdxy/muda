#include <list>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_buffer_2d.h>
#include <muda/buffer/device_buffer_3d.h>
#include <muda/buffer/reshape_nd/reserve.h>


namespace muda::details::buffer
{
template <typename BufferView>
class CopyConstructInfo
{
    using CBufferView = read_only_viewer<BufferView>;

  public:
    BufferView  dst;
    CBufferView src;
};

template <typename BufferView>
class ConstructInfo
{
  public:
    BufferView dst;
};

template <typename BufferView>
class DestructInfo
{
  public:
    BufferView dst;
};
}  // namespace muda::details::buffer


namespace muda
{
//template <typename T, typename FConstruct>
//void NDReshaper::resize(int              grid_dim,
//                        int              block_dim,
//                        cudaStream_t     stream,
//                        DeviceBuffer<T>& buffer,
//                        size_t           new_size,
//                        FConstruct&&     fct)
//{
//    auto& m_data     = buffer.m_data;
//    auto& m_size     = buffer.m_size;
//    auto& m_capacity = buffer.m_capacity;
//
//    if(new_size == m_size)
//        return;
//
//    auto old_size = m_size;
//
//    if(new_size < m_size)
//    {
//        // destruct the old memory
//        auto to_destruct = buffer.view(new_size, old_size - new_size);
//        details::buffer::kernel_destruct(grid_dim, block_dim, stream, to_destruct);
//        m_size = new_size;
//        return;
//    }
//
//    if(new_size <= m_capacity)
//    {
//        // construct the new memory
//        auto to_construct = buffer.view(old_size, new_size - old_size);
//        fct(to_construct);
//        m_size = new_size;
//    }
//    else
//    {
//        BufferView<T> dst = details::buffer::reserve_1d<T>(stream, new_size);
//
//        if(m_data)
//        {
//            // copy old data
//            details::buffer::kernel_assign(
//                grid_dim, block_dim, stream, dst, std::as_const(buffer).view(0, old_size));
//        }
//
//        // construct the new memory
//        {
//            BufferView<T> to_construct = dst.subview(old_size);
//            fct(to_construct);
//        }
//
//        if(m_data)
//            Memory(stream).free(m_data);
//
//        m_data     = dst.data();
//        m_size     = new_size;
//        m_capacity = new_size;
//    }
//}

// template <typename T, typename FConstruct>
MUDA_HOST void NDReshaper::resize(int                grid_dim,
                                  int                block_dim,
                                  cudaStream_t       stream,
                                  DeviceBuffer2D<T>& buffer,
                                  Extent2D           new_extent,
                                  FConstruct&&       fct)
{
    using namespace details::buffer;

    auto& m_data        = buffer.m_data;
    auto& m_pitch_bytes = buffer.m_pitch_bytes;
    auto& m_extent      = buffer.m_extent;
    auto& m_capacity    = buffer.m_capacity;

    if(new_extent == m_extent)
        return;


    auto old_extent = m_extent;

    std::list<CopyConstructInfo<Buffer2DView<T>>> copy_construct_infos;
    std::list<ConstructInfo<Buffer2DView<T>>>     construct_infos;
    std::list<DestructInfo<Buffer2DView<T>>>      destruct_infos;

    enum class Dimension
    {
        Width,
        Height
    };

    auto for_every_combination = []() {

    };

    if(new_extent < m_extent)
    {
        // if the new extent is smaller than the old extent in all dimensions
        // destruct the old memory
        m_extent = new_extent;
    }
    else if(new_extent <= m_capacity)
    {
        // all dimensions are bigger then the new extent
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
        Memory(stream).alloc_2d(&ptr,
                                &new_pitch_bytes,
                                sizeof(T) * new_capacity.width(),
                                new_capacity.height());

        // if the old buffer was allocated, copy old data
        if(m_data)
        {
            Buffer2DView<T> dst{ptr, new_pitch_bytes, Offset2D::Zero(), old_extent};
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
                Offset2D offset{old_extent.height(), 0};
                Buffer2DView<T> to_construct{ptr, new_pitch_bytes, offset, new_extent};
                fct(to_construct);
            }
            else if(old_extent.height() == new_extent.height())
            {
            }
        }

        m_data        = ptr;
        m_pitch_bytes = new_pitch_bytes;
        m_extent      = new_extent;
        m_capacity    = new_capacity;
    }








    // if the old buffer was allocated, deallocate it
    if(m_data)
        Memory(stream).free(m_data);
    return;
}
}  // namespace muda