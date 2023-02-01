#pragma once
#include "../GL/gl.h"
#include <GLFW/glfw3.h>

namespace muda
{
void process_input(GLFWwindow* window);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
}  // namespace muda
