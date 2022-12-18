# muda
muda is μ-Cuda, yet another painless cuda programming paradigm

```c++
#include <muda/muda.h>
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

## quick start

run example:

```shell
$ xmake f --example=true
$ xmake 
$ xmake run muda_example hello_muda
```
other examples:

```shell
$ xmake run muda_example -l
```
to show all examples.

play all examples

```shell
$ xmake run muda_example
```

## More Info

[zhihu-ZH](https://zhuanlan.zhihu.com/p/592439225) :  description of muda.





