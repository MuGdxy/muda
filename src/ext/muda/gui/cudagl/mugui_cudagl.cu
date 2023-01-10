/**
 * @file: ext/muda/gui/cudagl/mugui_cudagl.cu
 * @author: sailing-innocent
 * @create: 2023-01-09
 * @desp: the implementation of cuda-opengl interoplate
*/

// a gl buffer object could be registered using `cudaGraphicsGLRegister()`
// which returns a pointer to device memory
// also, a gl texture or renderbuffer object could be registered using `cudaGraphicsGLRegisterImage()` function
// which returns a CUDA array
// you can use your kernels to read from array by binding it to a texture or surface reference
// they can also be wrtten via
// `cudaGrpahicsRegisterFlagsSurfaceLoadStore()`
// `cudaMemcpy2D()`
// interal types are GL_RGBA_FLOAT32, GL_RGBA_8, GL_INTENSITY16 or GL_RGBA8UI

#include "mugui_cudagl.h"
#include <cuda_gl_interop.h>
#include "../utils/gl_utils.h"
#include <iostream>

namespace muda
{
// trans from left button corner coordinate to viewport center origin coordinate
__device__ float lr_corner_T_viewport_center(float u)
{
    return u * 2.0f - 1.0f;
}

__global__ void createVertices(float* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float u = ix / (float)width;
    float v = iy / (float)height;

    float x = lr_corner_T_viewport_center(u);
    float y = lr_corner_T_viewport_center(v);

    float freq = 18.0f;
    float w    = sinf(freq * sqrtf(x * x + y * y) - time * 6.0f);

    positions[8 * (iy * width + ix) + 0] = x;
    positions[8 * (iy * width + ix) + 1] = y;
    positions[8 * (iy * width + ix) + 2] = w;
    positions[8 * (iy * width + ix) + 3] = 1.0f;
    // generate color
    positions[8 * (iy * width + ix) + 4] = w * 2.0f;
    positions[8 * (iy * width + ix) + 5] = 0.3f;
    positions[8 * (iy * width + ix) + 6] = 0.8f;
    positions[8 * (iy * width + ix) + 7] = 1.0f;
}

void MuGuiCudaGL::init_buffers()
{
    if(m_vert_shader_path.length() > 0 && m_frag_shader_path.length() > 0)
    {
        m_shader = GLShader(m_vert_shader_path, m_frag_shader_path);
    }
    else
    {
        std::string default_vert_shader =
            "#version 450\n\
            layout(location = 0) in vec4 aPos;\n\
            layout(location = 1) in vec4     aColor;\n\
            out vec4 pColor;\n\
            void main()\n\
            {\n\
                gl_Position = aPos;\n\
                pColor      = aColor;\n\
            }";

        std::string default_frag_shader =
            "#version 450\n\
            in vec4  pColor;\n\
            out vec4 fragColor;\n\
            void main()\n\
            {\n\
                fragColor = pColor;\n\
            }";
        m_shader = GLShader();
        m_shader.compile(default_vert_shader, default_frag_shader);
    }


    // generate vertices array
    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);

    glGenBuffers(1, &m_positions_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions_VBO);

    unsigned int size = m_window_res[0] * m_window_res[1] * 8 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    // generate input: [point4, color4]
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSetDevice(0);  // set the device
    // register the gl buffer to cuda
    cudaGraphicsGLRegisterBuffer(&m_positionsVBO_CUDA, m_positions_VBO, cudaGraphicsMapFlagsWriteDiscard);
}

void MuGuiCudaGL::gen_vertices()
{
    float  time_value = static_cast<float>(glfwGetTime());
    float* positions;
    cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, m_positionsVBO_CUDA);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(m_window_res[0] / dimBlock.x, m_window_res[1] / dimBlock.y);
    createVertices<<<dimGrid, dimBlock>>>(
        positions, time_value, m_window_res[0], m_window_res[1]);
    cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA);
}

bool MuGuiCudaGL::frame()
{
    process_input(m_window);
    gen_vertices();

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(m_VAO);
    m_shader.use();
    glBindBuffer(GL_ARRAY_BUFFER, m_positions_VBO);
    glDrawArrays(GL_POINTS, 0, m_window_res[0] * m_window_res[1]);
    glfwSwapBuffers(m_window);
    glfwPollEvents();

    return glfwWindowShouldClose(m_window);
}

void MuGuiCudaGL::destroy_buffers()
{
    cudaGraphicsUnregisterResource(m_positionsVBO_CUDA);
    glDeleteBuffers(1, &m_positions_VBO);
}

MuGuiCudaGL::~MuGuiCudaGL()
{
    destroy_buffers();
    destroy_window();
}

}  // namespace muda
