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





