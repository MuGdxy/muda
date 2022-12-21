# muda
muda is μ-Cuda, yet another painless cuda programming paradigm

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

```cpp
#include <muda/muda.h>
using namespace muda;
int main()
{
	stream s;
    parallel_for(2/*gridDim*/, 32/*blockDim*/, 0/*sharedMem*/, s/*stream*/)
        .apply(4/*count*/, 
               [] __device__(int i) 
               { 
                   print("hello muda %d/4\n", i); 
               })
        .wait();
}
```

thread_only container

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

## More Info

[zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) :  description of muda.





