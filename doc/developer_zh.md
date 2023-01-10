# 开发者

[TOC]

## 源码目录

你可以在![](https://github.com/MuGdxy/muda) 获取muda的源码。使用git进行版本管理，main分支为发行分支，dev分支为开发分支。

根目录下有如下目录树：
- test: 测试目录，可以在下载后跑一遍
- test/playground: 开发测试目录，不会合并到main分支中
- 

## 开始开发

开启test option，以便进行功能测试。

```shell
$ xmake f --test=true
```


### 测试

测试框架`catch2`

```cpp
//file: my_test.cu
TEST_CASE("my_launch","[launch]")
{
    // my test code
}
```

[catch2 命令行](https://catch2.docsforge.com/v2.13.2/running/command-line/)

根据tag来运行测试

``` shell
$ xmake run muda_test [launch]
```

根据name来运行测试

```shell
$ xmake run muda_test my_launch
```

命名规则一般为 `TEST_CASE("功能名称"，"[功能标签]")`，例如`TEST_CASE("matrix wrapper","[blas]")`。

可以直接在`test/muda_test/`文件夹下调用：

```shell
$ py mk.py test_name test_tag
```

来生成对应的测试模板

### 提供例子

```shell
$ xmake f --example=true
```

依然使用`catch2`来提供例子，所有的例子命名依然如测试中所示，例如`TEST_CASE("hello_muda","[quick_start]")`

可以在`example/`文件夹下调用：

```shell
py mk.py example_name example_tag
```

来生成对应的例子模板。

例子格式

```cpp
#include <muda/muda.h>
#include <catch2/catch.hpp>
#include "../example_common.h" // to use example_desc
using namespace muda;

void muda_example()
{
    example_desc("this is an example for how to write a muda example.");// to print example description when play this example
}

TEST_CASE("example_name", "[example_tag]")
{
    muda_example();
}
```

### 启用全部选项

```shell
$ xmake f --dev=true
```

上面的命令等价于

```shell
$ xmake f --example=true --test=true --playground=true
```
## 模块

### 一些定义

- [muda-style kernel launch](#core-launch)

- [capture&mapping](#ext-buffer)

### 模块划分

muda

- core
  - launch: launch/parallel_for等muda-style kernel调用与graphNodeParms支持 [details](#core-launch)
  - viewer: 带边界检查的各种数据读取方式的总和 [details](#core-viewer)
  - graph: cuda graph支持 [details](#core-graph)
- ext
  - buffer: 内存申请/管理工具 [details](#ext-buffer)
  - algo: cuda CUB 的 wrapper [details](#ext-algo)
  - blas: cublas cusparse cusolver的wrapper [details](#ext-blas)
  - thread_only: stl-like thread only 容器库，[details](ext-thread only)
  - pba: physically based animation相关，例如碰撞检测等。[details](#ext-pba)
  - gui: debug gui tool 基本可视化工具, TODO [details](#ext-gui)

#### core-launch

##### 设计思想

muda-style launch是以lambda表达式和callable object为kernel实参的kernel调用方式，例如：

```cpp
//lambda
launch(1,1).apply(
    []__device()
    {
        print("hello muda!");
    });

//callable object
struct MyFunc
{
    __device__ void operator () () 
    {
        print("hello muda!");
    }
}

launch(1,1).apply(MyFunc());
```

核心实现方式为：

```cpp
template<typename F>
__global__ void genericKernel(F callable)
{
    callable();
}

template<typename F>
void apply(F callable)
{
    genericKernel<<<gridDim,blockDim,sharedMem,stream>>>(callable);
}
```

我们将callable object作为`genericKernel`的实参进行传入，以此实现muda-style的kernel调用。

对于开发者而言，可以设计的部分有：

```cpp
template<typename F>
__global__ void genericKernel(F callable, /*任何额外的参数，由apply调用时确定，并从这里传入*/)
{
    /*返回值可以被利用，但一般不需要*/ callable(/*可以要求用户定义符合传入参数的callable object*/);
    
    /*例如，可以要求用户传入的callable object符合签名：void (int i), i由genericKernel来传入*/
    /*这个部分可以参考parallel_for的实现，或者设计思想与资料1 */
}
```

参考[设计思想与资料1](#设计思想与资料)

#### core-viewer

##### 设计思想

viewer是对内存的访问方式，他具有如下特点：

1. 不具有内存的所有权
2. trivial copyable

2 的含义为，viewer的拷贝均为浅拷贝，这个拷贝能够在host/device之间自由的进行，没有任何的副作用

对于原生的cuda kernel而言，以下的情况非常常见：

```cpp
__global__ void RawKernel(T* data, int size)
{
    data[threadIdx.x] = 1;
}
```

进一步，将`T* data, int size`转化为`class indexer1D`，下面的实例代码忽略了构造函数：

```cpp
class indexer1D
{
private:
    T* data; 
    int size;
public:
    T& operator (int i /*logic index*/)() {return data[i/*exact data offset*/]; }
}
__global__ void RawKernel(indexer1D viewer)
{
    viewer(threadIdx.x) = 1;
}
```

这就是最简单的viewer的雏形。

对于更复杂的访问方式，我们需要实现更加复杂的viewer，例如CSR(稀疏矩阵 compressed sparse row format)等等。

为了解决访问越界问题，我们需要在viewer中实现对输入的全面检查。

```cpp
class indexer1D
{
private:
    T* data; 
    int size;
public:
    T& operator (int i /*logic index*/)() 
    {
        assert(i>=0 && i<size);
        return data[i/*exact data offset*/]; 
    }
}
```

一些类比：

常见的图形API中，Sampler/Memory View/Image View等概念和muda viewer的概念非常接近，可参考[设计思想与资料3](#设计思想与资料)

##### 实现注意事项

viewer的实现注意事项有：

1. 所有的输入必须有边界检查，若产生了错误操作，需要能够提供足够具体且明确的错误原因。
2. 构造函数只可包含裸data指针与边界信息（边界信息必须完整，哪怕他不参与计算offset）（例1）
3. 需要提供`make_xxx`函数用于从`universal/host/device_vector/device_buffer`等容器中快速生成viewer
4. 也可以对于特定的数据结构实现对应的`make_xxx`，例如`cse/csr`等就对特定的host端数据结构实现了对应的`make`函数
5. 对最适合的数据结构与viewer的映射，提供`make_viewer`作为默认映射（例2）



例1. idxer2D 是dense 2D array的viewer，`offset = x * dim_y + y`，虽然没有使用`dim_x`来计算`offset`，但边界信息包含`dim_x`，所以idxer2D的构造函数中必须出现`dim_x`（或者`Eigen::Vector2i`表示的`(dim_x, dim_y)`）

例2. idxer1D是dense 1D array的viewer，我们为`universal/host/device_vector/device_buffer`实现`make_idxer1D` , `make_idxer`,`make_viewer`用于生成idxer1D。



推荐的报错方式：

```c++
if constexpr(debugViewers)
    if(!(x >= 0 && x < dim_x))
        muda_kernel_error("mapper: out of range, index=(%d) dim_=(%d)\n", x, dim_x);
```

`muda_kernel_error`宏函数会自动填写thread/block/grid信息，并在`trapOnError==true`时trap掉当前kernel。

#### core-graph

[TODO]

#### ext-buffer

##### 内容

buffer模块中目前主要有三个部分：

1. device_buffer: 基于cudaXXXAsync实现的内存管理

2. container: device_var/host_var, device_vector/host_vector

3. composite: buffer的组合，用于实现特定的viewer对应的数据结构

其中，

1 为muda基于cuda C API cudaXXXAsync实现的，device端容器

2 中device_var/host_var为muda仿照thrust实现的，单变量容器，device_vector与host_vector为thrust原生容器。

除了容器的实现，1，2中还添加了方便生成viewer的函数`make_viewer/make_idexer`等等，为`capture&mapping`提供方便。

`capture&mapping`指的是：

```cpp
device_vector<int> vec;
launch(1,1)
    .apply(
    [
        // lambda的捕获列表中的下述行为被称为
        // capture&mapping
        vec_viewer = make_viewer(vec)
    ]
    __device__()
    {

    });
```

3 中将组合buffer中的一些容器，以构成一些常用的数据结构，并为之实现方便的`make_viewer`函数，以便`capture&mapping`

#### ext-algo

##### 内容

[TODO]

##### 实现注意实现

此模块主要内容为CUB Wrapper，例如你要对CUB DeviceScan算法进行封装，请在`algorithm/`文件夹下调用：

```shell
py mk.py DeviceScan
```

`mk.py`将会生成对应的文件，以便开发。

#### ext-blas

[TODO]

##### 实现注意事项

除了Wrapper以外，还需要设计Vector/Matrix的Viewer，用于处理Vector/Matrix切片，转置，共轭等操作，暂未开始实现。

#### ext-thread only

##### 内容

[TODO]

##### 实现注意事项

所有的容器与算法修改自EASTL，如需实现特定的容器，仅需将EASTL对应的函数前添加宏前缀`MUDA_THREAD_ONLY`即可（最内层的工具函数现已基本添加完毕，一般来说只要对最外层函数进行添加即可）。

部分可能不适合thread_only的内容可能需要酌情删除或修改。

#### allocator

thread only容器目前有三个allocator可使用：

| name                      | details                                                      |      |
| ------------------------- | ------------------------------------------------------------ | ---- |
| thread_allocator          | 通常的allocator，内存开辟于gpu global memory，不限制动态allocate数量 |      |
| thread_stack_allocator    | stack allocator，内存开辟于local memory，限制allocate数量    |      |
| external_buffer_allocator | 外部buffer allocator，内存开辟于指定的外部buffer中，一般用于shared memory或host端申请的gpu global memory上，限制allocate数量 |      |

若还有其他需求，可以根据`thread_allocator.h`进行添加。

#### ext-pba

[TODO]

#### ext-gui

[TODO]

### 其他实现注意事项

1. 请为自己实现的内容添加对应的test与example
2. 提交前请保证所有test/example均通过
3. 请在[implement.md](./implement.md)中填写对应的实现内容
4. 需要用户输入的测试请不要放入test中（以保证所有test可以一键启动并且执行到结束），可暂时放到playground中。
5. 避免使用64位类型和无符号类型，例如`int64_t/uint64_t`（无符号类型将可能触发GPU上的无符号数溢出检查）

## 设计思想与资料

1. launch/parallel_for: [小彭老师：CUDA在现代C++中如何运用(1:10:08处)](https://www.bilibili.com/video/BV16b4y1E74f/?spm_id_from=333.999.0.0&vd_source=4b0953be0f61d253c6c2f7ab9c4fa59f)

2. template: [C++ Templates - The Complete Guide, 2nd Edition](http://tmplbook.com/)
3. viewer：[Vulkan Buffer View/Image View](https://registry.khronos.org/vulkan/specs/1.3/html/chap12.html#resources-buffer-views)

4. muda 链式编程风格：[UniRx](https://github.com/neuecc/UniRx)
5. muda kernel launch风格：[CUB](https://nvlabs.github.io/cub/)
6. muda container: [thrust](https://docs.nvidia.com/cuda/thrust/index.html)

## 分支
- dev分支主要用来开发和提交pull-request