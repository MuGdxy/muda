#pragma once
/**
 * @file: ext/muda/gui/cudagl/mugui_cudagl.h
 * @author: sailing-innocent
 * @create: 2023-01-09
 * @desp: this is a cuda-opengl interop gui interface
*/
#include "../puregl/mugui_puregl.h"
#include "../utils/gl_shader.h"

namespace muda
{
class MuGuiCudaGL : public MuGuiPureGL
{
  public:
    MuGuiCudaGL() = default;
    MuGuiCudaGL(IMuGuiMode mode)
        : MuGuiPureGL(mode)
    {
    }
    ~MuGuiCudaGL();
    bool frame();
    void muda_gen_vertices(float* positions, float time, unsigned int width, unsigned int height);

  protected:
    void init_buffers() override;
    void destroy_buffers() override;

  protected:
    virtual void gen_vertices();  // the entry draw function

  protected:
    unsigned int           m_cuda_device = 0;
    unsigned int           m_positions_VBO;
    unsigned int           m_VAO;
    cudaGraphicsResource_t m_positionsVBO_CUDA;
};

}  // namespace muda