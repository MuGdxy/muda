#pragma once
#include <muda/launch/parallel_for.h>
#include <muda/ext/field/field_entry_view.h>
#include <muda/buffer/buffer_view.h>
#include <Eigen/Core>
namespace muda
{
class FieldEntryLaunch : public LaunchBase<FieldEntryLaunch>
{
  public:
    MUDA_GENERIC FieldEntryLaunch(cudaStream_t stream = nullptr)
        : LaunchBase(stream)
    {
    }

    /**********************************************************************************************
    * 
    * EntryView <- EntryView
    * 
    **********************************************************************************************/
    template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, 1, 1>  dst,
                        CFieldEntryView<T, SrcLayout, 1, 1> src);

    template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout, int N>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, N, 1>  dst,
                        CFieldEntryView<T, SrcLayout, N, 1> src);

    template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, M, N>  dst,
                        CFieldEntryView<T, SrcLayout, M, N> src);

    /**********************************************************************************************
    *   
    * EntryView <- Value
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout DstLayout>
    MUDA_HOST FieldEntryLaunch& fill(FieldEntryView<T, DstLayout, 1, 1> dst, const T& value);


    template <typename T, FieldEntryLayout DstLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& fill(FieldEntryView<T, DstLayout, M, N> dst,
                        const Eigen::Matrix<T, M, N>&      value);

    /**********************************************************************************************
    *   
    * BufferView <- EntryView
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout SrcLayout>
    MUDA_HOST FieldEntryLaunch& copy(BufferView<T> dst, CFieldEntryView<T, SrcLayout, 1, 1> src);


    template <typename T, FieldEntryLayout SrcLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(BufferView<Eigen::Matrix<T, M, N>>  dst,
                        CFieldEntryView<T, SrcLayout, M, N> src);

    /**********************************************************************************************
    *   
    * EntryView <- BufferView
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout DstLayout>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, 1, 1> dst, CBufferView<T> src);

    template <typename T, FieldEntryLayout DstLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, M, N>  dst,
                        CBufferView<Eigen::Matrix<T, M, N>> src);
};
}  // namespace muda

#include "details/field_entry_launch.inl"