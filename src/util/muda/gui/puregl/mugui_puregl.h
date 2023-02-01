#pragma once

/**
 * @file: ext/muda/gui/puregl/mugui_puregl.h
 * @author: sailing-innocent
 * @create: 2023-01-09
 * @desp: this is an pure opengl gui interface
*/

#include "../GL/gl.h"
#include "GLFW/glfw3.h"
#include "../mugui_base.h"
#include <string>
#include "../utils/gl_shader.h"

namespace muda
{
class MuGuiPureGL : public MuGuiBase
{
  public:
    MuGuiPureGL() = default;
    ~MuGuiPureGL();
    MuGuiPureGL(IMuGuiMode mode, std::string _vert_shader_path = "", std::string _frag_shader_path = "")
        : MuGuiBase(mode)
        , m_vert_shader_path{_vert_shader_path}
        , m_frag_shader_path{_frag_shader_path}
    {
    }
    void init(int resw = 800, int resh = 600) override;
    bool frame() override;

  protected:
    void         init_window(int resw, int resh) override;
    void         destroy_window() override;
    virtual void init_buffers(){};
    virtual void destroy_buffers(){};

  protected:
    std::string m_vert_shader_path = "";
    std::string m_frag_shader_path = "";
    GLFWwindow* m_window           = nullptr;
    GLShader    m_shader;
};
}  // namespace muda