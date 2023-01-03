# 开发者

[TOC]

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

### 定义

`muda-style kernel launch`：以lambda表达式和callable object为kernel实参的kernel调用方式。

### 模块划分

muda

- core
  - launch: launch/parallel_for等`muda-style` kernel 调用
  - viewer
  - 
- extension
  - PBA: physically based animation相关，例如碰撞检测等。
  - gui: debug gui tool 基本可视化工具, TODO。 

## 设计思想与资料

- launch/parallel_for: [小彭老师：CUDA在现代C++中如何运用(1:10:08处)](https://www.bilibili.com/video/BV16b4y1E74f/?spm_id_from=333.999.0.0&vd_source=4b0953be0f61d253c6c2f7ab9c4fa59f)
- template: [C++ Templates - The Complete Guide, 2nd Edition](http://tmplbook.com/)

## 各模块注意事项

1. 请为自己实现的内容添加对应的test与example
2. 提交前请保证所有test/example均通过
3. 请在[implement.md](./implement.md)中填写对应的实现内容
4. 避免使用64位类型和无符号类型，例如`int64_t/uint64_t`（无符号类型将可能触发GPU上的无符号数溢出检查）

### Viewer

Viewer的实现注意事项有：

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

### Algorithm

此模块主要内容为CUB Wrapper，例如你要对CUB DeviceScan算法进行封装，请在`algorithm/`文件夹下调用：

```shell
py mk.py DeviceScan
```

`mk.py`将会生成对应的文件，以便开发。

### Thread Only Container

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

### BLAS

除了Wrapper以外，还需要设计Vector/Matrix的Viewer，用于处理Vector/Matrix切片，转置，共轭等操作，暂未开始实现。
