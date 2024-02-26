[TOC]

# MUDA

MUDA is **μ-CUDA**, yet another painless CUDA programming **paradigm**.

> COVER THE LAST MILE OF CUDA

## Quick Overview

**Detailed Introduction And Overview [Highly Recommend]**  :arrow_right: https://mugdxy.github.io/muda-doc/ 

```c++
#include <muda/muda.h>
#include <muda/logger.h>
#include <iostream>
using namespace muda;

int main()
{
    constexpr int N = 8;
    
    // resizable buffer
    DeviceBuffer<int> buffer;
    buffer.resize(N);
    buffer.fill(1);
    
    // std::cout like logger
    Logger logger;
    
    // parallel for loop
    ParallelFor()
        .kernel_name("hello_muda") 
        .apply(N,
      	[
            buffer = buffer.viewer().name("buffer"),
            logger = logger.viewer()
        ] __device__(int i) 
        {
            logger << "buffer(" << i << ")=" << buffer(i) << "\n";
        });
    
    logger.retrieve(std::cout); // show print on std::cout
}
```

## Build

### Xmake

Run example:

```shell
$ xmake f --example=true
$ xmake 
$ xmake run muda_example hello_muda
```
To show all examples:

```shell
$ xmake run muda_example -l
```
Play all examples:

```shell
$ xmake run muda_example
```
### Cmake

```shell
$ mkdir CMakeBuild
$ cd CMakeBuild
$ cmake -S ..
$ cmake --build .
```

### Copy Headers

Because **muda** is header-only, copy the `src/muda/` folder to your project, set the include directory, and everything is done.

### Macro

| Macro                     | Value               | Details                                                      |
| ------------------------- | ------------------- | ------------------------------------------------------------ |
| `MUDA_CHECK_ON`           | `1`(default) or `0` | `MUDA_CHECK_ON=1` for turn on all muda runtime check(for safety) |
| `MUDA_WITH_COMPUTE_GRAPH` | `1`or`0`(default)   | `MUDA_WITH_COMPUTE_GRAPH=1` for turn on muda compute graph feature |

If you manually copy the header files, don't forget to define the macros yourself. If you use cmake or xmake, just set the project dependency to muda.

## Tutorial

- [tutorial_zh](https://zhuanlan.zhihu.com/p/659664377)
- If you need an English version tutorial, please get in touch with me or post an issue to let me know.

## Documentation

Documentation is maintained on https://mugdxy.github.io/muda-doc/. And you can also build the doc by yourself. 

## Examples

- [examples](./example/)

All examples in `muda/example` are self-explanatory,  enjoy it.

![image-20231102030703199](/docs/img/example-img.png)

## Contributing

Contributions are welcome. We are looking for or are working on:

1. **muda** development

2. fancy simulation demos using **muda**

3. better documentation of **muda**

## Related Work

- Topological braiding simulation using **muda** (old version)

  ```latex
  @article{article,
  author = {Lu, Xinyu and Bo, Pengbo and Wang, Linqin},
  year = {2023},
  month = {07},
  pages = {},
  title = {Real-Time 3D Topological Braiding Simulation with Penetration-Free Guarantee},
  volume = {164},
  journal = {Computer-Aided Design},
  doi = {10.1016/j.cad.2023.103594}
  }
  ```

  ![braiding](/docs/img/braiding.png)

  





