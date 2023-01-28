# muda
MUDA is **μ-CUDA**, yet another painless CUDA programming **paradigm**.

> stop bothering yourself using raw CUDA, just stay elegant and clear-minded.

## overview

easy launch:

```c++
#include <muda/muda.h>
using namespace muda;
int main()
{
    launch(1, 1)
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
    device_vector<int> dv(64, 1);
    stream             s;
    parallel_for(2, 16, 0, s) // parallel-semantic
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

## quick start

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
## tutorial

- [tutorial_zh](./doc/tutorial_zh.md)

## features

### thread_only container

allow you to use STL-like containers and algorithms in one thread.

```cpp
#include <muda/muda.h>
#include <muda/thread_only/priority_queue>
using namespace muda;
namespace to = muda::thread_only;
int main()
{
    launch(1, 1)
        .apply(
            [] __device__() mutable
            { 
                to::priority_queue<int> queue;
                auto& container = queue.get_container();
                container.reserve(16);
                queue.push(4);queue.push(5);queue.push(6);
                queue.push(7);queue.push(8);queue.push(9);
                while(!queue.empty())
                {
                    print("%d ", queue.top());
                    queue.pop();
                }
                //result: 9 8 7 6 5 4
            })
        .wait();
}
```

### graph support

a friendly way to create cuda graph.

```cpp
void graph_example()
{        
    device_var<int> value = 1;
    // create a graph
    auto            graph = graph::create();

    // create kernel as graph node parameters(we switch from .apply to .asNodeParms)
    auto pA = parallel_for(1).asNodeParms(1, 
        [] __device__(int i) mutable 
        { 
            print("A\n"); 
        });

    auto pB = parallel_for(1).asNodeParms(1, 
        [] __device__(int i) mutable 
        {
            print("B\n");
        });

    auto phB = host_call().asNodeParms(
        [] __host__() 
        {
            std::cout << "host" << std::endl; 
        });

    auto pC = parallel_for(1).asNodeParms(1,
        [value = make_viewer(value)] __device__(int i) mutable
        {
            print("C, value=%d\n", value);
            value = 2;
        });

    auto pD = launch(1, 1).asNodeParms(
        [value = make_viewer(value)] __device__() mutable
        { 
            print("D, value=%d\n", value); 
        });
	
    // create nodes and dependencies
    auto kA = graph->addKernelNode(pA);
    auto kB = graph->addKernelNode(pB);
    // hB deps on kA and kB
    auto hB = graph->addHostNode(phB, {kA, kB});
    // kC deps on hB
    auto kC = graph->addKernelNode(pC, {hB});
    // kD deps on kC
    auto kD = graph->addKernelNode(pD, {kC});
    
	// create instance (or say graphExec)
    auto instance = graph->instantiate();
    
    //launch on the default stream
    instance->launch();
}
```

## Contribute

Go to [developer_zh.md](./doc/developer_zh.md) and [zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) for further info.

## More Info

[zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) :  description of muda.





