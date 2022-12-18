# 开发者

## 开始开发

开启test option，以便进行功能测试。

```shell
$ xmake f --test=true
```

## 测试

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

可以直接向`test/muda_test`中添加`.cpp/.cu`文件进行快速测试。

## 提供例子

```shell
$ xmake f --example=true
```

依然使用`catch2`来提供例子，所有的例子命名依然如测试中所示，例如`TEST_CASE("hello_muda","[quick_start]")`

可以直接向`example/`中添加`.cpp/.cu`文件以创建例子。

例子格式

```cpp
#include <muda/muda.h>
#include <catch2/catch.hpp>
#include "../example_common.h"
using namespace muda;

void hello_muda()
{
    example_desc("this is an example for how to write an muda example");// to print example description when play this example
}

TEST_CASE("example_name", "[example_tag]")
{
    
}
```

