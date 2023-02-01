/**
 * @file: mugui_puregl.cpp
 * @author: sailing-innocent
 * @create: 2023-01-09
 * @desp: The Implementation for mugui pure gl
*/

#include "mugui_puregl.h"
#include <iostream>
#include "../GL/gl.h"
#include <GLFW/glfw3.h>
#include "../utils/gl_utils.h"

namespace muda
{
MuGuiPureGL::~MuGuiPureGL()
{
    destroy_buffers();
    destroy_window();
}
void MuGuiPureGL::init(int resw, int resh)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    init_window(resw, resh);
    init_buffers();
}

void MuGuiPureGL::init_window(int resw, int resh)
{
    m_window = glfwCreateWindow(resw, resh, "MuGui Pure GL Window", NULL, NULL);
    m_window_res[0] = resw;
    m_window_res[1] = resh;
    if(m_window == NULL)
    {
        std::cout << "failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

    if(!gladLoadGL(glfwGetProcAddress))
    {
        std::cout << "Failed to initiailze GLAD" << std::endl;
    }
}

void MuGuiPureGL::destroy_window()
{
    glfwTerminate();
}

bool MuGuiPureGL::frame()
{
    process_input(m_window);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(m_window);
    glfwPollEvents();

    return glfwWindowShouldClose(m_window);
}
}  // namespace muda
