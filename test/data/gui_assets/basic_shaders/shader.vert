#version 450

layout (location=0) in vec4 aPos;
layout (location=1) in vec4 aColor;

out vec4 pColor;

void main()
{
    gl_Position = aPos;
    pColor = aColor;
}