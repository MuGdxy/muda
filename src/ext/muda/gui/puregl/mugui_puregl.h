#include "../glad/gl.h"
#include "GLFW/glfw3.h"
#include "../mugui_base.h"

namespace muda
{
class MuGuiPureGL : public MuGuiBase
{
  public:
    MuGuiPureGL() = default;
    ~MuGuiPureGL();
    explicit MuGuiPureGL(IMuGuiMode mode)
        : MuGuiBase(mode)
    {
    }
    void init() override;
    void init_window(int resw, int resh) override;
    void destroy_window() override;
    bool frame() override;

  protected:
    bool begin_frame();
    void draw_gui();

  protected:
    GLFWwindow* m_window = nullptr;
};
}  // namespace muda