// don't place #pragma once at the beginning of this file
// because it should be inserted in multiple files

#define MUDA_CUB_WRAPPER_IMPL(x)                                               \
    size_t temp_storage_bytes = 0;                                             \
    void*  d_temp_storage     = nullptr;                                       \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    prepare_buffer(external_buffer, temp_storage_bytes);                       \
    d_temp_storage = (void*)external_buffer.data();                            \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    return *this;

#define MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(x)                                                        \
    ComputeGraphBuilder::invoke_phase_actions(                                                            \
        [&] { checkCudaErrors(x); },                                                                      \
        [&]                                                                                               \
        {                                                                                                 \
            MUDA_ASSERT(!ComputeGraphBuilder::is_building() || d_temp_storage != nullptr,                 \
                        "d_temp_storage must not be nullptr when building graph. you should not"          \
                        "query the temp_storage_size when building a compute graph, please do it outside" \
                        "a compute graph.");                                                              \
            ComputeGraphBuilder::capture([&](cudaStream_t)                                                \
                                         { checkCudaErrors(x); });                                        \
        });                                                                                               \
    return *this;