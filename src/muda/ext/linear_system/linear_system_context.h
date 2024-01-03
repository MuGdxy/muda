#pragma once
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <list>
#include <muda/buffer/device_buffer.h>
#include <muda/literal/unit.h>
#include <muda/mstl/span.h>

#include <muda/ext/linear_system/dense_vector_view.h>
#include <muda/ext/linear_system/dense_matrix_view.h>
namespace muda
{
class LinearSystemContextCreateInfo
{
  public:
    cudaStream_t stream = nullptr;
    // base size of temp buffer, if buffer is not enough
    // we create a new buffer with size = buffer_byte_size_base * 2 / 4 / 8 / 16 / ...
    // and we will not release the old buffer because of safety
    size_t buffer_byte_size_base = 256_M;
};
class LinearSystemContext
{
  private:
    cudaStream_t                       m_stream;
    cublasHandle_t                     m_cublas;
    cusparseHandle_t                   m_cusparse;
    cusolverDnHandle_t                 m_cusolver_dn;
    cusolverSpHandle_t                 m_cusolver_sp;
    std::list<DeviceBuffer<std::byte>> m_buffers;
    std::list<std::vector<std::byte>>  m_host_buffers;
    DeviceBuffer<std::byte>            m_scalar_buffer;

    LinearSystemContextCreateInfo    m_create_info;
    std::list<std::function<void()>> m_sync_callbacks;
    std::string                      m_current_label;
    bool                             m_pointer_mode_device = true;

  private:
    auto cublas() const { return m_cublas; }
    auto cusparse() const { return m_cusparse; }
    auto cusolver_dn() const { return m_cusolver_dn; }
    auto cusolver_sp() const { return m_cusolver_sp; }
    void set_pointer_mode_device();
    void set_pointer_mode_host();
    void shrink_temp_buffers();
    void add_sync_callback(std::function<void()>&& callback);

    BufferView<std::byte> temp_buffer(size_t size);

    span<std::byte> temp_host_buffer(size_t size);

    template <typename T>
    BufferView<T> temp_buffer(size_t size);

    template <typename T>
    span<T> temp_host_buffer(size_t size);

    template <typename T>
    std::vector<T*> temp_buffers(size_t size_in_buffer, size_t num_buffer);

    template <typename T>
    std::vector<T*> temp_host_buffers(size_t size_in_buffer, size_t num_buffer);


  public:
    LinearSystemContext(const LinearSystemContextCreateInfo& info = {});
    LinearSystemContext(const LinearSystemContext&)            = delete;
    LinearSystemContext& operator=(const LinearSystemContext&) = delete;
    LinearSystemContext(LinearSystemContext&&)                 = delete;
    LinearSystemContext& operator=(LinearSystemContext&&)      = delete;
    ~LinearSystemContext();

    void label(std::string_view label) { m_current_label = label; }
    auto label() const -> std::string_view { return m_current_label; }

    auto stream() const { return m_stream; }
    void stream(cudaStream_t stream);

    void sync();

  public:
    // TODO:
    // norm
    template <typename T>
    T norm(CDenseVectorView<T> x);
    template <typename T>
    void norm(CDenseVectorView<T> x, VarView<T> result);
    template <typename T>
    void norm(CDenseVectorView<T> x, T* result);
    // dot
    template <typename T>
    T dot(CDenseVectorView<T> x, CDenseVectorView<T> y);
    template <typename T>
    void dot(CDenseVectorView<T> x, CDenseVectorView<T> y, VarView<T> result);
    template <typename T>
    void dot(CDenseVectorView<T> x, CDenseVectorView<T> y, T* result);
    // axpby
    template <typename T>
    void axpby(const T& alpha, CDenseVectorView<T> x, const T& beta, DenseVectorView<T> y);
    template <typename T>
    void axpby(CVarView<T> alpha, CDenseVectorView<T> x, CVarView<T> beta, DenseVectorView<T> y);
    template <typename T>
    void plus(CDenseVectorView<T> x, CDenseVectorView<T> y, DenseVectorView<T> z);
    // spmv
    // mv
    template <typename T>
    void mv(CDenseMatrixView<T> A,
            const T&            alpha,
            CDenseVectorView<T> x,
            const T&            beta,
            DenseVectorView<T>  y);
    template <typename T>
    void mv(CDenseMatrixView<T> A,
            CVarView<T>         alpha,
            CDenseVectorView<T> x,
            CVarView<T>         beta,
            DenseVectorView<T>  y);
    template <typename T>
    void mv(CDenseMatrixView<T> A, CDenseVectorView<T> x, DenseVectorView<T> y);
    // solve
    // solve Ax = b, A will be modified for factorization
    // and b will be modified to store the solution
    template <typename T>
    void solve(DenseMatrixView<T> A_to_fact, DenseVectorView<T> b_to_x);
    // mm
};
}  // namespace muda

#include "details/linear_system_context.inl"
#include "details/routines/norm.inl"
#include "details/routines/dot.inl"
#include "details/routines/axpby.inl"
#include "details/routines/spmv.inl"
#include "details/routines/mv.inl"
#include "details/routines/solve.inl"
#include "details/routines/mm.inl"
