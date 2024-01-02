# MUDA
MUDA is **μ-CUDA**, yet another painless CUDA programming **paradigm**.

> COVER THE LAST MILE OF CUDA

## Overview

### Launch

Simple, self-explanatory, intellisense-friendly Launcher.

```c++
#include <muda/muda.h>
using namespace muda;
__global__ void raw_kernel()
{
    printf("hello muda!\n");
}

int main()
{
    // just launch
    Launch(1, 1)
        .apply(
        [] __device__() 
        {
            print("hello muda!\n"); 
        }).wait();
    
    constexpr int N = 8;
    // dynamic grid
    ParallelFor(256 /*block size*/)
        .apply(N,
        [] __device__(int i) 
        {
            print("hello muda %d!\n", i); 
        }).wait();
    
    // grid stride loop
    ParallelFor(8  /*grid size*/, 
                32 /*block size*/)
        .apply(N,
        [] __device__(int i) 
        {
            print("hello muda %d!\n", i); 
        }).wait();
    
    // intellisense-friendly wrapper 
    Kernel{raw_kernel}();
    Kernel{32/*grid size*/, 64/*block size*/,0/*shared memory*/, stream, other_kernel}(...)
    
    Stream stream;
    on(stream).wait();
}
```

### Logger

```c++
Logger logger;
Launch(2, 2)
    .apply(
        [logger = logger.viewer()] __device__() mutable
        {
            // type override
            logger << "int2: " << make_int2(1, 2) << "\n";
            logger << "float3: " << make_float3(1.0f, 2.0f, 3.0f) << "\n";
        })
    .wait();
// download the result to any ostream you like. 
logger.retrieve(std::cout);
```

### Buffer

```c++
DeviceBuffer<int> buffer;

// copy from std::vector
std::vector<int> host(8);
buffer.copy_from(host);

// copy to raw memory
int host_array[8];
buffer.copy_to(host_array);

// use BufferView to copy sub-buffer
buffer.view(0,4).copy_from(host.data());

DeviceBuffer<int> dst_buffer{4};
// use BufferView to copy sub-buffer
buffer.view(0,4).copy_to(dst_buffer.view());

// safe and easy resize
DeviceBuffer2D<int> buffer2d;
buffer.resize(Extent2D{5, 5}, 1);
buffer.resize(Extent2D{7, 2}, 2);
buffer.resize(Extent2D{2, 7}, 3);
buffer.resize(Extent2D{9, 9}, 4);
// subview 
buffer2d.view(Offset2D{1,1}, Extent2D{3,3});
buffer2d.copy_to(host);

DeviceBuffer3D<int> buffer3d;
buffer3d.resize(Extent3D{3, 4, 5}, 1);
buffer3d.copy_to(host);
```

result of buffer2d:

![buffer2d_resize](README.assets/buffer2d_resize.svg)

### Viewer In Kernel

```c++
DeviceVar<int> single;
DeviceBuffer<int> array;
DeviceBuffer2D<int> array2d;
DeviceBuffer3D<int> array3d;
Logger logger;
Launch().apply(
[
    single  = single.viewer().name("single"), // give a name for more readable debug info
    array   = buffer.viewer().name("array"),
    array2d = buffer_2d.viewer().name("array2d"),
    array3d = buffer_3d.viewer().name("array3d"),
    logger  = logger.viewer().name("logger"),
    ...
] __device__ () mutable
{
    single = 1;
    array(i) = 1;
    array2d(offset_in_height, offset_in_width) = 1;
    array3d(offset_in_depth, offset_in_height, offset_in_width) = 1;
    logger << 1;
});
```

### Event And Stream

```c++
Stream         s1, s2;
Event          set_value_done;

DeviceVar<int> v = 1;
on(s1)
    .next<Launch>(1, 1)
    .apply(
        [v = v.viewer()] __device__() mutable
        {
            int next = 2;
            v = next;
        })
    .record(set_value_done)
    .apply(
        [] __device__()
        {
            some_work();
        });

on(s2)
    .when(set_value_done)
    .next<Launch>(1, 1)
    .apply([v = v.viewer()] __device__()
           { int res = v; });
```

### Asynchronous Operation

```c++
// kernel launch
Kernel{..., f}(...);
Launch(stream).apply(...).wait();
ParallelFor(stream).apply(N, ...).wait();

// graph launch
GraphLaunch().launch(graph).wait();

// Memory
Memory(stream).copy(...).wait();
Memory(stream).set(...).wait();

// Buffer: for BufferView/Buffer2DView/Buffer3DView
BufferLaunch(stream).copy(BufferView, ...).wait();
BufferLaunch(stream).fill(BufferView,...).wait();
```

###  Field Layout

A simple simulation code using `muda::Field`, with a `muda::eigen` extension for better performance and readability.

```c++
// create a field
Field field;
// create a subfield called "particle"
// all attribute in this subfield has the same size
auto& particle = field["particle"];
// create a logger
Logger logger;

// build the subfield
auto  builder = particle.SoAoS(); // use SoAoS layout
auto& m       = builder.entry("mass").scalar<float>();
auto& pos     = builder.entry("position").vector3<float>(); 
auto& vel     = builder.entry("velocity").vector3<float>();
auto& f       = builder.entry("force").vector3<float>();
builder.build(); // finish building a subfield

// set size of the particle attributes
constexpr int N = 10;
particle.resize(N);

// to use muda eigen extension
using namespace Eigen;
using namespace muda::eigen;

ParallelFor(256)
    .apply(N,
           [m   = make_viewer(m),          // muda::eigen::make_viewer
            pos = make_viewer(pos),        // muda::eigen::make_viewer
            vel = make_viewer(vel),        // muda::eigen::make_viewer
            f   = make_viewer(f)] $(int i) // syntax sugar for `__device__ (int i) mutable`
           {
               m(i)   = 1.0f;
               pos(i) = Vector3f::Zero();
               vel(i) = Vector3f::Zero();
               f(i)   = Vector3f{0.0f, -9.8f, 0.0f}; // just gravity
           })
    .wait();

// safely resize the subfield
particle.resize(N * 13);

float dt = 0.01f;

// integration
ParallelFor(256)
    .apply(N,
           [logger = logger.viewer(),
            m   = make_viewer(m),
            pos = make_viewer(pos),
            vel = make_viewer(vel),
            f   = make_cviewer(f),
            dt] $(int i)
           {
               auto     x = pos(i);
               auto     v = vel(i);
               Vector3f a = f(i) / m(i);
               v = v + a * dt;
               x = x + v * dt;
               logger << "position=" << x << "\n"; 
           })
    .wait();

logger.retrieve(std::cout);
```

Note: every entry can be a separate `muda::ComputeGraphVar` so that it can also be used in `muda::ComputeGraph`

### Compute Graph

**MUDA** can generate `cudaGraph` nodes and dependencies from your `eval()` call. And the `cudaGraphExec` will be automatically updated (minimally) if you update a `muda::ComputeGraphVar`. More details in [zhihu_ZH](https://zhuanlan.zhihu.com/p/658080362).

Define a muda compute graph:

```c++
void compute_graph_simple()
{
    ComputeGraphVarManager manager;
    ComputeGraph graph{manager};

    // 1) define GraphVars
    auto& N   = manager.create_var<size_t>("N");
    // BufferView represents a fixed range of memory
    // dynamic memory allocation is not allowed in GraphVars
    auto& x_0 = manager.create_var<BufferView<Vector3>>("x_0");
    auto& x   = manager.create_var<BufferView<Vector3>>("x");
    auto& y   = manager.create_var<BufferView<Vector3>>("y");
    
    // 2) create GraphNode
    graph.create_node("cal_x_0") << [&]
    {
        // initialize values
        ParallelFor(256).apply(N.eval(),
                               [x_0 = x_0.eval().viewer()] __device__(int i) mutable
                               { x_0(i) = Vector3::Ones(); });
    };

    graph.create_node("copy_to_x") // copy
        << [&] { BufferLaunch().copy(x.eval(), x_0.ceval()); };

    graph.create_node("copy_to_y") // copy
        << [&] { BufferLaunch().copy(y.eval(), x_0.ceval()); };

    graph.create_node("print_x_y") << [&]
    {
        // print
        ParallelFor(256).apply(N.eval(),
                               [x = x.ceval().cviewer(),
                                y = y.ceval().cviewer(),
                                N = N.eval()] __device__(int i) mutable
                               {
                                   if(N <= 10)
                                       print("[%d] x = (%f,%f,%f) y = (%f,%f,%f) \n",
                                             i,
                                             x(i).x(),
                                             x(i).y(),
                                             x(i).z(),
                                             y(i).x(),
                                             y(i).y(),
                                             y(i).z());
                               });
    };
    // 3) visualize it using graphviz (for debug)
    graph.graphviz(std::cout);
}
```

![graphviz](README.assets/compute_graph.svg)

Launch a muda compute graph:

```c++
void compute_graph_simple()
{
    // resources
    auto N_value    = 4;
    auto x_0_buffer = DeviceVector<Vector3>(N_value);
    auto x_buffer   = DeviceVector<Vector3>(N_value);
    auto y_buffer   = DeviceVector<Vector3>(N_value);

    N.update(N_value);
    x_0.update(x_0_buffer);
    x.update(x_buffer);
    y.update(y_buffer);
    
    // create stream
    Stream stream;
    // sync graph on stream
    graph.launch(stream);
    // launch all nodes on a single stream (fallback to origin cuda kernel launch)
    graph.launch(true, stream);
}
```

### Dynamic Parallelism

```c++
void dynamic_parallelism_graph()
{
    std::vector<int> host(16);
    std::iota(host.begin(), host.end(), 0);

    ComputeGraphVarManager manager;
    // create graph
    ComputeGraph      graph{manager, "graph", ComputeGraphFlag::DeviceLaunch};
    // create resource
    DeviceBuffer<int> src = host;
    DeviceBuffer<int> dst(host.size());
	
    // create graph var
    auto& src_var = manager.create_var("src", src.view());
    auto& dst_var = manager.create_var("dst", dst.view());
	
    // create graph node
    graph.$node("copy")
    {
        BufferLaunch().copy(dst_var, src_var);
    };
    // build graph
    graph.build();
	
    // create a scheduler graph
    ComputeGraph launch_graph{manager, "launch_graph", ComputeGraphFlag::DeviceLaunch};
    auto& graph_var = manager.create_var("graph", graph.viewer());
	
    // create a node to launch our graph
    launch_graph.$node("launch")
    {
        Launch().apply(
            [graph = graph_var.ceval()] $()
            {
                graph.tail_launch();
            });
    };
    // graphviz all graph we created
    manager.graphviz(std::cout);
    // launch and wait
    launch_graph.launch().wait();
}
```

![image-20231119161837442](README.assets/dynamic_parallelism.svg)

### MUDA vs. CUDA

```c++
/* 
* muda style
*/
void muda()
{
    DeviceBuffer<int>  dv(64);
    dv.fill(1);
    
    ParallelFor(256) // parallel-semantic
        .kernel_name("my_kernel") // or just .kernel_name(__FUNCTION__)
        .apply(64, // automatically cover the range
               [
                   // mapping from the DeviceBuffer to a proper viewer
                   // which can be trivially copy through device and host
                   dv = dv.viewer().name("dv")
               ] 
               __device__(int i) mutable
               { 
                   dv(i) *= 2; // safe, the viewer check the boundary automatically
               })
        .wait();// happy waiting, muda remember the stream.
    	//.apply(...) //if you want to go forward with the same config, just call .apply() again.
}


/* 
* cuda style
*/

// manually create kernel
__global__ void times2(int* i, int N) // modifying parameters is horrible
{
    auto tid = threadIdx.x;
    if(tid < N) // check corner case manaully
    {
        i[tid] *= 2;// unsafe: no boundary check at all
    }
}

void cuda()
{
    // to be brief, we just use thrust to allocate memory
    thrust::device_vector<int> dv(64, 1);
    // cast to raw pointer
    auto dvptr = thrust::raw_pointer_cast(dv.data());
    // create stream and check error
    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    // call the kernel (which always ruins the Intellisense, if you use VS.)
    times2<<<1, 64, 0, s>>>(dvptr, dv.size());
    // boring waiting and error checking
    checkCudaErrors(cudaStreamSynchronize(s));
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

Because **muda** is header-only, just copy the `src/muda/` folder to your project, set the include directory, and then everything is done.

### Macro

| Macro                     | Value               | Details                                                      |
| ------------------------- | ------------------- | ------------------------------------------------------------ |
| `MUDA_CHECK_ON`           | `1`(default) or `0` | `MUDA_CHECK_ON=1` for turn on all muda runtime check(for safety) |
| `MUDA_WITH_COMPUTE_GRAPH` | `1`or`0`(default)   | `MUDA_WITH_COMPUTE_GRAPH=1` for turn on muda compute graph feature |

If you manually copy the header files, don't forget to define the macros yourself. If you use cmake or xmake, just set the project dependency to muda.

## Tutorial

- [tutorial_zh](https://zhuanlan.zhihu.com/p/659664377)
- If you need an English version tutorial, please contact me or post an issue to let me know.

## Examples

- [examples](./example/)

All examples in `muda/example` are self-explanatory,  enjoy it.

![image-20231102030703199](README.assets/example-img.png)

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

  ![braiding](README.assets/braiding.png)

  





