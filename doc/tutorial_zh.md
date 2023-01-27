[TOC]

# MUDA 使用教程
## MUDA 是什么
MUDA全称μ-CUDA，是基于CUDA的一种编程范式，旨在帮助用户更优雅更安全地使用CUDA。

MUDA核心模块主要有：
-  launch：muda-style kernel调用
-  viewer：内存观察器
-  graph：CUDA Graph生成支持

在接下来的章节中我们会对这些模块以及使用进行详细的介绍。

## 引入 MUDA
使用MUDA可以通过以下两种方式：
1. [包管理安装](https://github.com/MuGdxy/muda-app)
2. [源码引入](./import_guide.md)

第一种方式引入MUDA将会非常方便。
第二种方式给予用户更完整的控制权。

在第一种方式的链接中，我们准备了一个muda的开始程序，你可以从这个程序开始你的muda使用之旅。
如果你不太熟悉xmake的使用方式，可以查阅xmake官方文档 https://xmake.io/。不必担心，使用xmake时，我们会对每一个语句进行解释，以便读者理解。

### IDE 支持

如果你是visual studio用户，可以在命令行下键入以下内容，以生成对应的sln文件，此后一切工作均由vs接管。
```shell
$ xmake project -k vsxmake2022
```
## Note

术语定义内容将由斜体表示，例如：

*定义：muda是一种cuda的编程范式*

本Tutorial会在段落中增加补充内容，如下表示：

> 这是额外的内容

对额外内容不熟悉或者不理解并不会影响后续阅读，可以大胆略过。

## Quick Start

### hello muda

打开muda-app/src/main.cu
我们可以看到如下代码：

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
            })
    	.wait();
}
```

在muda中launch类是一个最简单的kernel调用器，他的配置方式和cuda的`<<<>>>`方式几乎一致，只是将尖括号中的参数移动到了launch的构造函数中。

> 目前为止visual studio的intellisense都无法识别三尖括号，使用三尖括号会导致代码提示使用体验一言难尽，muda直接避免了这种情况。

kernel的调用从apply函数正式开始。和原生cuda kernel调用不同的是，muda的kernel调用均要求传入的对象为`__device__` [callable object](https://devtut.github.io/cpp/callable-objects.html#function-pointers)

> 因为muda内部将通过一些`__global__`kernel函数来调用用户传入的callable object。即，所有的用户kernel都是经muda kernel函数**代理**的。

apply后，我们使用一个`wait()`函数对当前的cuda stream(default stream)进行等待。

> 在这个例子中，我们能够非常直观的感受到一种链式编程的范式，很像响应式（Reactive）编程范式，例如[UniRx](https://github.com/neuecc/UniRx)。
>
> 响应式编程在实践中被证明是比较适合异步操作的编程范式。当然如果你不了解响应式编程也完全没有关系，这不会影响你使用muda。

*定义：muda-style launch是一种使用链式编程与device callable object进行kernel调用的调用方式*

### vector add

OK！我们来做一个简单的事情，记得CUDA的[VectorAdd](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/vectorAdd/vectorAdd.cu)的例子吗？

Cuda VectorAdd例子计算了两个向量的和，并且把他们相加得到和向量。

接下来，我们使用muda来实现相同的功能。

```c++
#include <muda/muda.h>
using namespace muda;
int main()
{
	constexpr int N = 50000;
    // alloc host and device data
    host_vector<float>   hA(N), hB(N), hC(N);
    device_vector<float> dA(N), dB(N), dC(N);

    // initialize A and B using random numbers
    auto rand = [] { return std::rand() / (float)RAND_MAX; };
    std::generate(hA.begin(), hA.end(), rand);
    std::generate(hB.begin(), hB.end(), rand);

    // copy A and B to device
    dA = hA;
    dB = hB;
	
    int threadsPerBlock = 256;
  	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    launch(blockPerGrid, threadsPerBlock)
        .apply(
        [
            // this is a capture list
            // here, we do the mapping from device_vector to a viewer
            // which can be used in kernel (much safer than raw pointer)
            dC = make_viewer(dC),  
            dA = make_viewer(dA),  
            dB = make_viewer(dB),
            N = N
        ]__device__() mutable  // place "mutable" to make dC modifiable
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < numElements) dC(i) = dA(i) + dB(i);
        })
        .wait();  // wait the kernel to finish

    // copy C back to host
    hC = dC;
}
```

与hello muda相比，我们的lambda表达式的捕获列表中出现了`make_viewer`函数，`make_viewer`函数能够从host端的容器、裸指针等生成为一个device中可以访问对应内存的观察者（viewer）。

在上述例子中，`make_viewer`将`device_vector<float>`映射为了`dense1D<float>`（一种muda-viewer），便于我们在kernel中用`dC(i)`这样的方式来访问我们想要的元素。

> 常见的viewer有：
>
> - dense1D，如同一维数组一样访问内存。`dense1D(i)`
> - dense2D，如同二维数组一样访问内存。`dense2D(i,j)`
> - dense3D, 如同三维数组一样访问内存。`dense3D(i,j,k)`
> - csr, compressed sparse row/column (稀疏矩阵行/列压缩格式)
> - cse, compressed sparse elements (一种稀疏二维数据结构压缩格式)
>
> 最常用的一般就是 dense1D。

*定义：muda-viewer（或缩写为viewer) 是一类内存的观察器，只提供访问内存的方式而不具有内存的所有权。*

> viewer的特性：
>
> 1. 复制viewer没有任何副作用，不论是在host端还是在device端还是在host/device端相互复制
> 2. viewer将逻辑索引（logical index）转化为内存的偏移（memory offset）
> 3. viewer会处理逻辑索引输入并进行边界检查，避免一切越界行为与空指针访问等内存问题

目前阶段，我们要清楚，如果想要在muda-style kernel中使用预先申请的内存，就必须通过viewer作为媒介。



上面代码并不那么的优雅，我们发现，并行代码的逻辑似乎和kernel的调用参数产生了紧密的联系，这使得代码的可读性大幅下降。其实，像向量相加这样并行遍历内存的情况非常普遍，所以，muda提供了一个新的launcher——parallel_for，来抽象这样的调用。

我们将launch相关的代码换成parall_for。

```c++
parallel_for(256)
    .apply(N, // parallel for count
           [dC = make_viewer(dC),
            dA = make_viewer(dA),
            dB = make_viewer(dB)] 
           __device__(int i) mutable
           {
               // safe parallel_for will cover the rang [0, N)
               // i just goes from 0 to N-1
               dC(i) = dA(i) + dB(i);
           })
    .wait();  // wait the kernel to finish
```

可以发现，我们并没有手动去计算究竟需要多少的gridDim，因为这个计算已经由parall_for为我们完成了，我们只需要关心需要遍历的总数量`N`即可，并且在kernel中，我们也没有任何边界检查的代码，因为parallel_for会为我们剔除这些情况。

诶？那如果我们想用固定的gridDim和blockDim来实现遍历怎么办呢？parallel_for当然考虑到了这一点，我们只需要指定gridDim和blockDim的大小，parall_for就可以以[grid stride loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)的方式进行遍历。我们的核心代码并不需要做任何修改。

```cpp
parallel_for(32 /*gridDim*/,64/*blockDim*/)
    .apply(N, // parallel for count
           [dC = make_viewer(dC),
            dA = make_viewer(dA),
            dB = make_viewer(dB)] 
           __device__(int i) mutable
           {
               // safe parallel_for will cover the rang [0, N)
               // i just goes from 0 to N-1
               dC(i) = dA(i) + dB(i);
           })
    .wait();  // wait the kernel to finish
```

当然，除了可以指定遍历的大小`N`你还可以用一下形式来指定遍历的范围

- (count) : 遍历区间[0, count)
- (begin, count)：遍历区间 [begin, begin + count)
- (begin, end, step): 以step为步长遍历区间[begin, end]

> 注：
>
> muda将内存的所有权和内存的访问两者显式地分离开，内存的申请释放工作应当交给：
>
> - device_vector
> - device_var
> - device_buffer
> - 手动 memory alloc/free，来管理。
> 
> 而各类内存的访问应当交给各种viewer来实现，二者是解耦的。
>
> 因为同一块内存可能有多种访问方式，不同访问方式有不同的逻辑索引，在不同的逻辑索引下，有不同的边界要求。
>
> 一个简单的例子就是：
>
> 对于一块`int buffer[32]`而言，`dense1D(dim_i=32)`的访问方式认为，当索引$i \ge 32$或$i<0$时，存在越界。
>
> 而对于`dense2D(dim_i=2, dim_j=16)`而言，当索引$i\ge 2$或$i<0$或$j \ge 16$或$j<0$时，存在越界。
>
> 当我们使用`dense2D(0,17)`进行访问时，尽管对于`buffer`而言并没有越界，但对于逻辑索引而言确实有越界产生。
>
> 这是muda要求所有权与访问方式分离的原因之一。

体验越界：

现在我们将代码改成如下形式：

```c++
parallel_for(32 /*gridDim*/,64/*blockDim*/)
    .apply(N, // parallel for count
           [dC = make_viewer(dC),
            dA = make_viewer(dA),
            dB = make_viewer(dB)] 
           __device__(int i) mutable
           {
               // safe parallel_for will cover the rang [0, N)
               // i just goes from 0 to N-1
               dC(i + 1) = dA(i + 1) + dB(i + 1);
           })
    .wait();  // wait the kernel to finish
```

重新运行，你会得到一个准确的报错信息。



[TODO] : 需要在这个位置粘贴报错的结果



报错信息中，

`()-()`括号内的为，`(blockId|gridDim) - (threadId|blockDim)`

`viewer[unnamed]`中的unnamed是因为我们没有对正在使用的viewer进行命名。我们可以通过下面的方式来对viewer进行命名。

```c++
parallel_for(32 /*gridDim*/,64/*blockDim*/)
    .apply(N, // parallel for count
           [dC = make_viewer(dC).name("dC"),
            dA = make_viewer(dA).name("dA"),
            dB = make_viewer(dB).name("dB")] 
           __device__(int i) mutable
           {
               // safe parallel_for will cover the rang [0, N)
               // i just goes from 0 to N-1
               dC(i + 1) = dA(i + 1) + dB(i + 1);
           })
    .wait();  // wait the kernel to finish
```

命名可以发生在任何区域，host端/device端/capture&mapping区，上面的代码中，我们的命名就发生在capture&mapping区，这也是最方便，最具可读性的方式。

```c++
int main()
{
    device_vector<int> v;
    viewer = make_viewer(v).name("host");
    launch(1,1).apply([viewer = make_viewer(v).name("capture_mapping")]__device__()
                      {
                          viewer.name("device");
                      });
}
```

*定义：capture&mapping区是muda-style launch中的lambda表达式捕获列表区域，此区域一般用于make_viewer与复制值类型数据到device*

实际上，在debug模式下的muda允许用户设置viewer的名称，并用于错误输出，非debug模式下，muda将会无视命名函数。

### synchronization

[TODO] : stream-wait / event-record-when-wait的使用方法

### syntactic sugar

[TODO] :  on/next 语法

```c++
on(s1).next<launch>(1, 1)
```

### container and buffer

[TODO] : device_vector, device_var, device_buffer的使用方法

### bindless

[TODO] : 使用viewer of viewer与kernel new来实现bindless

## GUI

[TODO] : 

## Graph

[TODO] : CUDA Graph支持与基于资源的依赖图生成工具graph Manager

## Thread Only Container

[TODO] : thread only container与algorithm

## Cub Wrapper

[TODO] : muda对cub库的支持

## Blas Wrapper

[TODO] :  muda对基本线性代数库的支持

## Other

### Physically Based Animation Utils

[TODO] : 使用muda实现的基于物理的动画工具库
