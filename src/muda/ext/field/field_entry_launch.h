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
    template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(FieldEntryView<T, DstLayout, M, N>  dst,
                                     CFieldEntryView<T, SrcLayout, M, N> src);

    /**********************************************************************************************
    *   
    * EntryView <- Value
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout DstLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& fill(FieldEntryView<T, DstLayout, M, N> dst,
                                     const typename FieldEntryView<T, DstLayout, M, N>::ElementType& value);

    /**********************************************************************************************
    *   
    * BufferView <- EntryView
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout SrcLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(
        BufferView<typename CFieldEntryView<T, SrcLayout, M, N>::ElementType> dst,
        CFieldEntryView<T, SrcLayout, M, N> src);

    /**********************************************************************************************
    *   
    * EntryView <- BufferView
    *   
    * *********************************************************************************************/
    template <typename T, FieldEntryLayout DstLayout, int M, int N>
    MUDA_HOST FieldEntryLaunch& copy(
        FieldEntryView<T, DstLayout, M, N> dst,
        CBufferView<typename FieldEntryView<T, DstLayout, M, N>::ElementType> src);
};
}  // namespace muda

#include "details/field_entry_launch.inl"