#pragma once
/**
 * @file: ext/muda/gui/mugui_base.h
 * @author: sailing-innocent
 * @create: 2023-01-09
 * @desp: the header file for muda-gui base
*/


namespace muda
{
enum IMuGuiMode
{
    RaytraceMesh
};

class MuGuiBase
{
  public:
    MuGuiBase() = default;
    explicit MuGuiBase(IMuGuiMode mode)
        : m_gui_mode(mode)
    {
    }
    virtual ~MuGuiBase() {}
    virtual void init()                          = 0;
    virtual void init_window(int resw, int resh) = 0;
    virtual void destroy_window()                = 0;
    virtual bool frame()                         = 0;

  protected:
    IMuGuiMode m_gui_mode;
    int        m_window_res[2];
    bool       m_render_window = true;
    bool       m_gui_redraw    = true;
};
}  // namespace muda