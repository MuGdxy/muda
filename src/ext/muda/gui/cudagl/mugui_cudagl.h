#include "muda/gui/puregl/mugui_puregl.h"

namespace muda
{
class MuGuiCudaGL : public MuGuiPureGL
{
  public:
    MuGuiCudaGL();
    MuGuiCudaGL(IMuGuiMode mode)
        : MuGuiPureGL(mode)
    {
    }
    ~MuGuiCudaGL();
    bool frame();

  protected:
    unsigned int                 m_cuda_device = 0;
    unsigned int                 m_positions_VBO;
    unsigned int                 m_VAO;
    struct cudaGraphicsResource* m_positionsVBO_CUDA;
};

}  // namespace muda