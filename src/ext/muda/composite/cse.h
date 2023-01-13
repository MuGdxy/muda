#pragma once
#include <muda/viewer/cse.h>
#include <muda/container/vector.h>
#include <muda/buffer/device_buffer.h>

namespace muda
{
template <typename DataContainer, typename BeginContainer, typename CountContainer>
class compressed_sparse_elements
{
  public:
    DataContainer  data;
    BeginContainer begin;
    CountContainer count;
    compressed_sparse_elements() = default;
    compressed_sparse_elements(int data_count, int dim_i) noexcept
    {
        this->data_count(data_count);
        this->dim_i(dim_i);
    }
    void data_count(int size) noexcept { data.resize(size); }
    int  data_count() const noexcept { return data.size(); }

    void dim_i(int size) noexcept
    {
        begin.resize(size);
        count.resize(size);
    }
    int dim_i() const noexcept
    {
        if constexpr(DEBUG_COMPOSITE)
        {
            if(begin.size() != count.size())
                throw std::logic_error("begin and count must have the same size");
        }
        return begin.size();
    }
    
    compressed_sparse_elements& operator=(compressed_sparse_elements rhs) {
        if(this == &rhs)
            return *this;
        data  = rhs.data;
        begin = rhs.begin;
        count = rhs.count;
        return *this;
    }
    
    template <typename OtherDataContainer, typename OtherBeginContainer, typename OtherCountContainer>
    compressed_sparse_elements& operator=(
        compressed_sparse_elements<OtherDataContainer, OtherBeginContainer, OtherCountContainer>& rhs)
    {
        data  = rhs.data;
        begin = rhs.begin;
        count = rhs.count;
        return *this;
    }
};

template <typename T>
using device_cse =
    compressed_sparse_elements<device_vector<T>, device_vector<int>, device_vector<int>>;

template <typename T>
using host_cse =
    compressed_sparse_elements<host_vector<T>, host_vector<int>, host_vector<int>>;

template <typename T>
using universal_cse =
    compressed_sparse_elements<universal_vector<T>, universal_vector<int>, universal_vector<int>>;

template <typename T>
class device_buffer_cse
    : public compressed_sparse_elements<device_buffer<T>, device_buffer<int>, device_buffer<int>>
{
  public:
    device_buffer_cse() = default;

    device_buffer_cse(int data_count, int dim_i) noexcept
        : compressed_sparse_elements<device_buffer<T>, device_buffer<int>, device_buffer<int>>(
            data_count, dim_i)
    {
    }

    void stream(cudaStream_t s) noexcept
    {
        this->data.stream(s);
        this->begin.stream(s);
        this->count.stream(s);
    }

    cudaStream_t stream() const noexcept
    {
        if constexpr(DEBUG_COMPOSITE)
        {
            if(!(this->data.stream() == this->begin.stream()
                 && this->begin.stream() == this->count.stream()))
                throw std::logic_error("data/begin/count should has the same stream");
        }
        return this->data.stream();
    }
};
}  // namespace muda

namespace muda
{
template <typename DataContainer, typename BeginContainer, typename CountContainer>
inline __host__ auto make_cse(
    compressed_sparse_elements<DataContainer, BeginContainer, CountContainer>& cse) noexcept
{
    if constexpr(DEBUG_COMPOSITE)
    {
        if(cse.begin.size() != cse.count.size())
            throw std::logic_error("begin and count must have the same size");
    }
    return muda::cse<typename DataContainer::value_type>(
        data(cse.data), cse.data_count(), data(cse.begin), data(cse.count), cse.dim_i());
}
template <typename DataContainer, typename BeginContainer, typename CountContainer>
inline __host__ auto make_viewer(compressed_sparse_elements<DataContainer, BeginContainer, CountContainer>& cse)
{
    return make_cse(cse);
}
}  // namespace muda