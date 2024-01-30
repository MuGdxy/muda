[TOC]

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

![buffer2d_resize](/docs/img/buffer2d_resize.svg)

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

### [Extension] Linear System Support

[We are still working on this part]

**MUDA** supports basic linear system operations. e.g.:

1. Sparse Matrix Format Conversion
2. Sparse Matrix Assembly
3. Linear System Solving

![](./docs/img/linear_system.drawio.svg)

The only thing you need to do is to declare a `muda::LinearSystemContext`.

```c++
LinearSystemContext ctx;
// non-unique triplets of (row_index, col_index, block3x3)
DeviceTripletMatrix<float, 3> A_triplet;
// setup the triplet matrix dimension
A_triplet.reshape(block_rows,block_cols);
// resize the triplets, we should know the total count of the triplets.
A_triplet.resize_triplets(hessian_count); 

// unique triplets of (row_index, col_index, block3x3) faster SPMV than TripletMatrix
DeviceBCOOMatrix<float,3> A_bcoo; 

// block compressed sparse row format, faster SPMV than BCOOMatrix
DeviceBSRMatrix<float,3> A_bsr; 

// compressed sparse row format, slower SPMV than BSRMatrix
DeviceCSRMatrix<float,3> A_csr

// trivial dense matrix 
DeviceDenseMatrix<float> A_dense; 

// convert:
ctx.convert(A_triplet, A_bcoo);
ctx.convert(A_bcoo, A_bsr);
ctx.convert(A_bsr, A_dense);
ctx.convert(A_bsr, A_csr);
ctx.convert(A_bcoo, A_dense);

// so for the Sparse Vector ...
```

We only allow users to assemble a Sparse Matrix from Triplet Matrix. And allow users to read from BCOOMatrix.

To assemble a Triplet Matrix, user need to use the `viewer` of a Triplet Matrix.

```c++
DeviceTripletMatrix<float, 3> A_triplet;
A_triplet.resize(block_rows,block_cols,hessian_count);
DeviceDenseVector<float> x, b;
x.resize(block_rows * 3);
b.resize(block_rows * 3)

ParallelFor(256/*block size*/)
    .apply(hessian_count,
    [
        H = A_triplet.viewer().name("Hessian"),
        g = x.viewer().name("gradient")
        // some infos to build up hessian and gradient
    ] __device__(int i) 
    {
        int row, col;
        Eigen::Matrix3f hessian; // fill the local hessian, using your infos
        Eigen::Vector3f gradient;
        
        // write the (row, col, hessian) to the i-th triplet
        H(i).write(row,col, hessin);
        
        // atomic add the gradient vector
        g.segment<3>(i * 3).atomic_add(gradient);
    }).wait();

// convert to bcoo for better performance on SPMV.
ctx.convert(A_triplet, A_bcoo);
ctx.convert(A_bcoo, A_bsr);

// maybe in some iterative solver:
ctx.spmv(A_bsr.cview(), x.cview(), b.view());
```

### [Extension] Field Layout

```c++
#include <muda/ext/field.h> // all you need for muda::Field

void field_example(FieldEntryLayout layout)
{
    using namespace muda;
    using namespace Eigen;

    Field field;
    // create a subfield called "particle"
    // any entry in this field has the same size
    auto& particle = field["particle"];
    float dt       = 0.01f;

    // build the field:
    // auto builder = particle.AoSoA(); // compile time layout
    auto builder = particle.builder(FieldEntryLayout::AoSoA);  // runtime layout
    auto& m      = builder.entry("mass").scalar<float>();
    auto& pos    = builder.entry("position").vector3<float>();
    auto& pos_old = builder.entry("position_old").vector3<float>();
    auto& vel     = builder.entry("velocity").vector3<float>();
    auto& force   = builder.entry("force").vector3<float>();
    // matrix is also supported, but in this example we don't use it
    auto& I = builder.entry("inertia").matrix3x3<float>();
    builder.build();  // finish building the field

    // set size of the particle attributes
    constexpr int N = 10;
    particle.resize(N);

    Logger logger;

    ParallelFor(256)
        .kernel_name("setup_vars")
        .apply(N,
               [logger = logger.viewer(),
                m      = m.viewer(),
                pos    = pos.viewer(),
                vel    = vel.viewer(),
                f      = force.viewer()] $(int i)
               {
                   m(i)   = 1.0f;
                   pos(i) = Vector3f::Ones();
                   vel(i) = Vector3f::Zero();
                   f(i)   = Vector3f{0.0f, -9.8f, 0.0f};

                   logger << "--------------------------------\n"
                          << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "pos=" << pos(i) << "\n"
                          << "vel=" << vel(i) << "\n"
                          << "f=" << f(i) << "\n";
               })
        .wait();

    logger.retrieve();

    // safe resize, the data will be copied to the new buffer.
    // here we just show the possibility
    // later we only work on the first N particles
    particle.resize(N * 2);

    ParallelFor(256)
        .kernel_name("integration")
        .apply(N,
               [logger = logger.viewer(),
                m      = m.cviewer(),
                pos    = pos.viewer(),
                vel    = vel.viewer(),
                f      = force.cviewer(),
                dt] $(int i)
               {
                   auto     x = pos(i);
                   auto     v = vel(i);
                   Vector3f a = f(i) / m(i);

                   v = v + a * dt;
                   x = x + v * dt;

                   logger << "--------------------------------\n"
                          << "i=" << i << "\n"
                          << "m=" << m(i) << "\n"
                          << "pos=" << pos(i) << "\n"
                          << "vel=" << vel(i) << "\n"
                          << "f=" << f(i) << "\n";
               })
        .wait();

    logger.retrieve();

    // copy between entry and host
    std::vector<Vector3f> positions;
    pos.copy_to(positions);
    pos.copy_from(positions);

    // copy between entries
    pos_old.copy_from(pos);

    // copy between buffer and entry
    DeviceBuffer<Vector3f> pos_buf;
    pos.copy_to(pos_buf);
    pos.copy_from(pos_buf);
}
```

Note that every `FieldEntry` has a `View` called `FieldEntryView`. A `FieldEntryView` can be regarded as a `ComputeGraphVar`(see below), which means `FieldEntry` can also be used in `ComputeGraph`. 

### Compute Graph

Define `MUDA_WITH_COMPUTE_GRAPH`  to turn on `Compute Graph` support.

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

![graphviz](/docs/img/compute_graph.svg)

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

![image-20231119161837442](/docs/img/dynamic_parallelism.svg)

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

Because **muda** is header-only, copy the `src/muda/` folder to your project, set the include directory, and everything is done.

### Macro

| Macro                     | Value               | Details                                                      |
| ------------------------- | ------------------- | ------------------------------------------------------------ |
| `MUDA_CHECK_ON`           | `1`(default) or `0` | `MUDA_CHECK_ON=1` for turn on all muda runtime check(for safety) |
| `MUDA_WITH_COMPUTE_GRAPH` | `1`or`0`(default)   | `MUDA_WITH_COMPUTE_GRAPH=1` for turn on muda compute graph feature |

If you manually copy the header files, don't forget to define the macros yourself. If you use cmake or xmake, just set the project dependency to muda.

## Tutorial

- [tutorial_zh](https://zhuanlan.zhihu.com/p/659664377)
- If you need an English version tutorial, please contact me or post an issue to let me know.

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

  





