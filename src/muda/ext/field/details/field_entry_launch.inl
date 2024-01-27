namespace muda
{
//using T                  = float;
//constexpr auto M         = 3;
//constexpr auto N         = 3;
//constexpr auto DstLayout = FieldEntryLayout::RuntimeLayout;
//constexpr auto SrcLayout = FieldEntryLayout::RuntimeLayout;

template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(FieldEntryView<T, DstLayout, 1, 1> dst,
                                                   CFieldEntryView<T, SrcLayout, 1, 1> src)
{
    MUDA_ASSERT(dst.count() == src.count(),
                "FieldEntry size mismatching: dst.count() = %d, src.count() = %d",
                dst.count(),
                src.count());
    ParallelFor().apply(dst.count(),
                        [=] __device__(int i) mutable
                        { *dst.data(i) = *src.data(i); });
    return *this;
}

template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout, int N>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(FieldEntryView<T, DstLayout, N, 1> dst,
                                                   CFieldEntryView<T, SrcLayout, N, 1> src)
{
    MUDA_ASSERT(dst.count() == src.count(),
                "FieldEntry size mismatching: dst.count() = %d, src.count() = %d",
                dst.count(),
                src.count());
    ParallelFor().apply(dst.count(),
                        [=] __device__(int i) mutable
                        {
#pragma unroll
                            for(int j = 0; j < N; ++j)
                            {
                                *dst.data(i, j) = *src.data(i, j);
                            }
                        });
    return *this;
}

template <typename T, FieldEntryLayout DstLayout, FieldEntryLayout SrcLayout, int M, int N>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(FieldEntryView<T, DstLayout, M, N> dst,
                                                   CFieldEntryView<T, SrcLayout, M, N> src)
{
    MUDA_ASSERT(dst.count() == src.count(),
                "FieldEntry size mismatching: dst.count() = %d, src.count() = %d",
                dst.count(),
                src.count());
    ParallelFor().apply(dst.count(),
                        [=] __device__(int i) mutable
                        {
#pragma unroll
                            for(int j = 0; j < M; ++j)
                            {
#pragma unroll
                                for(int k = 0; k < N; ++k)
                                {
                                    *dst.data(i, j, k) = *src.data(i, j, k);
                                }
                            }
                        });

    return *this;
}


template <typename T, FieldEntryLayout DstLayout>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::fill(FieldEntryView<T, DstLayout, 1, 1> dst,
                                                   const T& value)
{

    ParallelFor().apply(dst.count(),
                        [=] __device__(int i) mutable { *dst.data(i) = value; });

    return *this;
}

template <typename T, FieldEntryLayout DstLayout, int M, int N>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::fill(FieldEntryView<T, DstLayout, M, N> dst,
                                                   const Eigen::Matrix<T, M, N>& value)
{
    ParallelFor().apply(dst.count(),
                        [dst, value] __device__(int i) mutable
                        {
                            if constexpr(N == 1)
                            {
#pragma unroll
                                for(int j = 0; j < M; ++j)
                                {
                                    *dst.data(i, j) = value(j);
                                }
                            }
                            else
                            {
#pragma unroll
                                for(int j = 0; j < M; ++j)
                                {
#pragma unroll
                                    for(int k = 0; k < N; ++k)
                                    {
                                        *dst.data(i, j, k) = value(j, k);
                                    }
                                }
                            }
                        });

    return *this;
}

template <typename T, FieldEntryLayout SrcLayout>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(BufferView<T> dst,
                                                   CFieldEntryView<T, SrcLayout, 1, 1> src)
{
    MUDA_ASSERT(dst.size() == src.count(),
                "FieldEntry size mismatching: dst.size() = %d, src.count() = %d",
                dst.size(),
                src.count());

    ParallelFor()  //
        .apply(dst.size(),
               [=] __device__(int i) mutable { *dst.data(i) = *src.data(i); });

    return *this;
}

template <typename T, FieldEntryLayout SrcLayout, int M, int N>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(BufferView<Eigen::Matrix<T, M, N>> dst,
                                                   CFieldEntryView<T, SrcLayout, M, N> src)
{
    MUDA_ASSERT(dst.size() == src.count(),
                "FieldEntry size mismatching: dst.size() = %d, src.count() = %d",
                dst.size(),
                src.count());


    ParallelFor()  //
        .apply(dst.size(),
               [dst, src] __device__(int i) mutable
               {
                   if constexpr(N == 1)
                   {
#pragma unroll
                       for(int j = 0; j < M; ++j)
                       {
                           (*dst.data(i))(j) = *src.data(i, j);
                       }
                   }
                   else
                   {


#pragma unroll
                       for(int j = 0; j < M; ++j)
                       {
#pragma unroll
                           for(int k = 0; k < N; ++k)
                           {
                               (*dst.data(i))(j, k) = *src.data(i, j, k);
                           }
                       }
                   }
               });

    return *this;
}

template <typename T, FieldEntryLayout DstLayout>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(FieldEntryView<T, DstLayout, 1, 1> dst,
                                                   CBufferView<T> src)
{
    MUDA_ASSERT(dst.count() == src.size(),
                "FieldEntry size mismatching: dst.count() = %d, src.size() = %d",
                dst.count(),
                src.size());

    ParallelFor().apply(dst.count(),
                        [=] __device__(int i) mutable
                        { *dst.data(i) = *src.data(i); });
    return *this;
}

template <typename T, FieldEntryLayout DstLayout, int M, int N>
MUDA_HOST FieldEntryLaunch& FieldEntryLaunch::copy(FieldEntryView<T, DstLayout, M, N> dst,
                                                   CBufferView<Eigen::Matrix<T, M, N>> src)
{
    MUDA_ASSERT(dst.count() == src.size(),
                "FieldEntry size mismatching: dst.count() = %d, src.size() = %d",
                dst.count(),
                src.size());

    ParallelFor().apply(dst.count(),
                        [dst, src] __device__(int i) mutable
                        {
                            if constexpr(N == 1)
                            {
#pragma unroll
                                for(int j = 0; j < M; ++j)
                                {
                                    *dst.data(i, j) = (*src.data(i))(j);
                                }
                            }
                            else
                            {
#pragma unroll
                                for(int j = 0; j < M; ++j)
                                {
#pragma unroll
                                    for(int k = 0; k < N; ++k)
                                    {
                                        *dst.data(i, j, k) = (*src.data(i))(j, k);
                                    }
                                }
                            }
                        });

    return *this;
}
}  // namespace muda