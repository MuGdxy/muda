# muda
MUDA is **μ-CUDA**, yet another painless CUDA programming **paradigm**.

> stop bothering yourself using raw CUDA, just stay elegant and clear-minded.

## overview

### direct launch

easy launch:

```c++
#include <muda/muda.h>
using namespace muda;
int main()
{
    Launch(1, 1)
        .apply(
        [] __device__() 
        {
            print("hello muda!\n"); 
        }).wait();
}
```

muda vs cuda:

```c++
/* 
* muda style
*/
void muda()
{
    DeviceVector<int> dv(64, 1);
    stream             s;
    ParallelFor(2, 16, 0, s) // parallel-semantic
        .apply(64, //automatically cover the range using (gridim=2, blockdim=16)
               [
                   // mapping from the device_vector to a proper viewer
                   // which can be trivially copy through device and host
                   dv = make_viewer(dv) 
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

void muda_vs_cuda()
{
    // to be brief, we just use thrust to allocate memory
    thrust::device_vector<int> dv(64, 1);
    // cast to raw pointer
    auto                       dvptr = thrust::raw_pointer_cast(dv.data());
    // create stream and check error
    cudaStream_t               s;
    checkCudaErrors(cudaStreamCreate(&s));
    // call the kernel (which always ruins the Intellisense, if you use VS.)
    times2<<<1, 64, 0, s>>>(dvptr, dv.size());
    // boring waiting and error checking
    checkCudaErrors(cudaStreamSynchronize(s));
}
```

### auto compute graph

**muda** can generate `cudaGraph` nodes and dependencies from your `eval()` call. And the cuda `graphExec` will be automatically updated (minimally) if you update a `muda::ComputeGraphVar`. 

```c++
void compute_graph_example()
{    
	// resources
    size_t                N = 10;
    DeviceVector<Vector3> x_0(N);
    DeviceVector<Vector3> x(N);
    DeviceVector<Vector3> v(N);
    DeviceVector<Vector3> dt(N);
    DeviceVar<float>      toi;
    HostVector<Vector3>   h_x(N);

    // define graph vars
    ComputeGraph graph;

    auto& var_x_0 = graph.create_var<Dense1D<Vector3>>("x_0", make_viewer(x_0));
    auto& var_h_x = graph.create_var<Dense1D<Vector3>>("h_x", make_viewer(h_x));
    auto& var_x   = graph.create_var<Dense1D<Vector3>>("x", make_viewer(x));
    auto& var_v   = graph.create_var<Dense1D<Vector3>>("v", make_viewer(v));
    auto& var_toi = graph.create_var<Dense<float>>("toi", make_viewer(toi));
    auto& var_dt  = graph.create_var<float>("dt", 0.1);
    auto& var_N   = graph.create_var<size_t>("N", N);

    // define graph nodes
    graph.create_node("cal_v") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [v = var_v.eval(), dt = var_dt.eval()] __device__(int i) mutable
                   {
                       // simple set
                       v(i) = Vector3::Ones() * dt * dt;
                   });
    };

    graph.create_node("cd") << [&]
    {
        ParallelFor(256)  //
            .apply(var_N,
                   [x   = var_x.ceval(),
                    v   = var_v.ceval(),
                    dt  = var_dt.eval(),
                    toi = var_toi.ceval()] __device__(int i) mutable
                   {
                       // maybe some collision detection
                   });
    };

    graph.create_node("cal_x") << [&]
    {
        ParallelFor(256).apply(var_N.eval(),
                               [x   = var_x.eval(),
                                v   = var_v.ceval(),
                                dt  = var_dt.eval(),
                                toi = var_toi.ceval()] __device__(int i) mutable
                               {
                                   // integrate
                                   x(i) += v(i) * toi * dt;
                               });
    };

    graph.create_node("transfer") << [&]
    {
        Memory().transfer(var_x_0.eval().data(), var_x.ceval().data(), N * sizeof(Vector3));
    };

    graph.create_node("download") << [&]
    {
        Memory().download(var_h_x.eval().data(), var_x.ceval().data(), N * sizeof(Vector3));
    };
	
    // print graphviz to visualize the compute graph
    graph.graphviz(std::cout);
    
    // launch all nones of this graph in one stream (rollback to launch kernels on a single stream)
    // it is useful to debug or profile the graph.
    graph.launch(true);
    
    // update graph var
    var_N.update(5);
    // before launch, all related graph nodes will be updated
    graph.launch();
}
```

![graphviz](README.assets/graphviz.svg)

## build

### xmake

run example:

```shell
$ xmake f --example=true
$ xmake 
$ xmake run muda_example hello_muda
```
to show all examples:

```shell
$ xmake run muda_example -l
```
play all examples:

```shell
$ xmake run muda_example
```
### cmake

```shell
$ mkdir CMakeBuild
$ cd CMakeBuild
$ cmake -S ..
$ cmake --build .
```

### copy header

Because muda is header-only, so just copy the `src/muda/` folder to your project, set the include directory, and then everything is done.

### macro

| Macro                         | Value                                 | Details                                                      |
| ----------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `MUDA_CHECK_ON`               | `1` or `0`                            | `MUDA_CHECK_ON=1` for turn on all muda runtime check(for safety) |
| `MUDA_VIEWER_NAME_MAX_LENGTH` | `16`(default) or any positive integer | Every muda viewer has a name field(`char name[]`) , which will be passed with the viewer. |

## tutorial

- [tutorial_zh](./doc/tutorial_zh.md)

## Contribute

Go to [developer_zh.md](./doc/developer_zh.md) and [zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) for further info.

## More Info

[zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) :  description of muda.





