# The GUI Developing Guide

the gui system of cuda exists in `src/util/muda/gui`


## Playground Test

You can have a test on playground now with a cuda-opengl interop example:

```shell
$ xmake f --dev=true
$ xmake
$ xmake run muda_pg cuda_gl
```

## Linux

You have to install opengl support for your linux distro

for example: `sudo apt-get install mesa-common-dev`

