namespace muda
{
MUDA_INLINE SubFieldInterface::~SubFieldInterface()
{
    if(m_data_buffer)
        Memory().free(m_data_buffer).wait();
};

template <typename F>
void SubFieldInterface::resize_data_buffer(size_t size, F&& func)
{
    if(m_data_buffer == nullptr)
    {
        Memory().alloc(&m_data_buffer, size).set(m_data_buffer, size, 0).wait();
        m_data_buffer_size = size;
    }
    else if(size > m_data_buffer_size)
    {
        auto old_ptr       = m_data_buffer;
        auto old_size      = m_data_buffer_size;
        m_data_buffer_size = size;
        Memory().alloc(&m_data_buffer, size);
        func(old_ptr, old_size, m_data_buffer, size);
        Memory().free(old_ptr).wait();
    }
}

MUDA_INLINE void SubFieldInterface::copy_resize_data_buffer(size_t size)
{
    resize_data_buffer(size,
                       [](std::byte* old_ptr, size_t old_size, std::byte* new_ptr, size_t new_size)
                       {
                           Memory()
                               .set(new_ptr + old_size, new_size - old_size, 0)  // set the new memory to 0
                               .transfer(new_ptr, old_ptr, old_size);  // copy the old memory to the new memory
                       });
}

MUDA_INLINE uint32_t SubFieldInterface::round_up(uint32_t x, uint32_t n)
{
    MUDA_ASSERT((n & (n - 1)) == 0, "n is not power of 2");
    return (x + n - 1) & ~(n - 1);
}

MUDA_INLINE uint32_t SubFieldInterface::align(uint32_t offset,
                                              uint32_t size,
                                              uint32_t min_alignment,
                                              uint32_t max_alignment)
{
    auto alignment = std::clamp(size, min_alignment, max_alignment);
    return round_up(offset, alignment);
}
}  // namespace muda